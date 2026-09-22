"""Async serving engine: requests queue up, a single worker packs them into
prefill batches under a token budget, runs each batch in a thread (torch
releases the GIL) and resolves every request's future.

There is no decode phase, so a batch finishes in one forward pass and the
worker simply takes whatever is waiting: FIFO up to ``max_batch_tokens``,
waiting at most ``max_wait_ms`` for company when the queue is short.
"""
import asyncio
import time
from dataclasses import dataclass, field

from .readout import answers


class Overloaded(RuntimeError):
    pass


@dataclass
class Stats:
    requests: int = 0
    batches: int = 0
    tokens: int = 0
    busy_s: float = 0.0
    latency_s: float = 0.0
    errors: int = 0
    started: float = field(default_factory=time.time)


class AsyncEngine:
    def __init__(self, planner, runner, max_batch_tokens=8192, max_wait_ms=2.0, max_queue=1024):
        self.planner = planner
        self.runner = runner
        self.max_batch_tokens = max_batch_tokens
        self.max_wait = max_wait_ms / 1000
        self.max_queue = max_queue
        self.queue = asyncio.Queue()
        self.stats = Stats()
        self._worker = None

    def start(self):
        if self._worker is None:
            self._worker = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self):
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def decide(self, state, questions):
        """Answers keyed by question id, and the input tokens the read took."""
        plan = self.planner.plan(state, questions)  # raises SchemaError on bad requests
        if not plan.branches:
            return answers(plan, []), 0
        if self.queue.qsize() >= self.max_queue:
            raise Overloaded("dan is at capacity. Retry shortly.")
        self.start()
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((plan, fut, time.perf_counter()))
        return await fut

    @staticmethod
    def cost(plan):
        return len(plan.prefix) + sum(len(b.suffix) for b in plan.branches)

    async def _loop(self):
        while True:
            batch = [await self.queue.get()]
            budget = self.cost(batch[0][0])
            deadline = time.perf_counter() + self.max_wait
            while budget < self.max_batch_tokens:
                if self.queue.empty():
                    left = deadline - time.perf_counter()
                    if left <= 0:
                        break
                    await asyncio.sleep(min(left, 0.0005))
                    continue
                plan = self.queue._queue[0][0]  # peek: stop before the budget is exceeded
                if budget + self.cost(plan) > self.max_batch_tokens:
                    break
                batch.append(self.queue.get_nowait())
                budget += self.cost(plan)
            await self._run(batch, budget)

    async def _run(self, batch, tokens):
        plans = [p for p, _, _ in batch]
        t = time.perf_counter()
        try:
            logits = await asyncio.to_thread(self.runner.read_many, plans)
        except Exception as e:  # noqa: BLE001 - fail this batch's requests, keep serving
            self.stats.errors += len(batch)
            for _, fut, _ in batch:
                if not fut.done():
                    fut.set_exception(e)
            return
        done = time.perf_counter()
        self.stats.batches += 1
        self.stats.tokens += tokens
        self.stats.busy_s += done - t
        for (plan, fut, t0), lg in zip(batch, logits):
            self.stats.requests += 1
            self.stats.latency_s += done - t0
            if not fut.done():
                fut.set_result((answers(plan, lg), self.cost(plan)))
