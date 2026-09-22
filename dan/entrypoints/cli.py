"""`dan serve MODEL` — run the Jev-compatible server."""
import argparse
import os

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_engine(args):
    from transformers import AutoTokenizer

    from ..engine import AsyncEngine
    from ..prompt import Planner
    from ..runner import Runner

    runner = Runner(args.model, args.device, DTYPES[args.dtype], cache_bytes=int(args.prefix_cache_gb * (1 << 30)))
    return AsyncEngine(Planner(AutoTokenizer.from_pretrained(args.model)), runner,
                       max_batch_tokens=args.max_batch_tokens, max_wait_ms=args.max_wait_ms, max_queue=args.max_queue)


def serve(args):
    import uvicorn

    from .api_server import ServerSettings, create_app

    if args.threads:
        torch.set_num_threads(args.threads)
    settings = ServerSettings(args.model, args.served_model_name, args.api_key or os.environ.get("DAN_API_KEY", ""))
    uvicorn.run(create_app(settings, build_engine(args)), host=args.host, port=args.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dan")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve a model over the Jev-compatible HTTP API")
    s.add_argument("model", help="Hugging Face model id or local path")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--dtype", choices=DTYPES, default="bfloat16" if torch.cuda.is_available() else "float32")
    s.add_argument("--served-model-name", default="dan-latest")
    s.add_argument("--api-key", default="", help="require this bearer token (or set DAN_API_KEY)")
    s.add_argument("--max-batch-tokens", type=int, default=8192)
    s.add_argument("--max-wait-ms", type=float, default=2.0)
    s.add_argument("--max-queue", type=int, default=1024)
    s.add_argument("--prefix-cache-gb", type=float, default=2.0)
    s.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = default)")
    s.set_defaults(func=serve)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
