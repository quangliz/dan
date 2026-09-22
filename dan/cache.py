"""Keys and values of static prompt prefixes (instructions + question schema).

Requests that share a question schema share the prompt up to the state, so
its keys and values are computed once and reused. Entries are keyed by the
exact token ids, so a hit is always exact; eviction is least recently used
under a byte budget.
"""
from collections import OrderedDict


class PrefixCache:
    def __init__(self, max_bytes=2 << 30):
        self.max_bytes = max_bytes
        self.used = 0
        self.entries = OrderedDict()  # tuple(ids) -> (per-layer [(k, v)], bytes)
        self.hits = self.misses = 0

    def get(self, ids):
        key = tuple(ids)
        hit = self.entries.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return hit[0]

    def put(self, ids, kv):
        key = tuple(ids)
        if key in self.entries:
            return
        size = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)
        if size > self.max_bytes:
            return
        while self.used + size > self.max_bytes:
            _, (_, freed) = self.entries.popitem(last=False)
            self.used -= freed
        self.entries[key] = (kv, size)
        self.used += size
