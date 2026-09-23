"""dan command line.

    dan serve MODEL [--profile NAME=profile.json] [--adapter DIR]
    dan eval MODEL DATA.jsonl [--profile profile.json] [--adapter DIR]
    dan calibrate MODEL FIT.jsonl [--eval EVAL.jsonl] --out profile.json
    dan train MODEL DATA.jsonl --out adapter_dir
"""
import argparse
import json
import os

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_engine(args):
    from transformers import AutoTokenizer

    from ..engine import AsyncEngine
    from ..prompt import Planner
    from ..runner import Runner

    runner = Runner(args.model, args.device, DTYPES[args.dtype], cache_bytes=int(args.prefix_cache_gb * (1 << 30)),
                    adapter=args.adapter, quantization=args.quantization)
    return AsyncEngine(Planner(AutoTokenizer.from_pretrained(args.model)), runner,
                       max_batch_tokens=args.max_batch_tokens, max_wait_ms=args.max_wait_ms, max_queue=args.max_queue)


def load_profiles(specs):
    from ..calibration import Profile

    out = {}
    for s in specs or []:
        name, sep, path = s.partition("=")
        if not sep:
            raise SystemExit(f"--profile takes NAME=PATH, got {s!r}")
        out[name] = Profile.load(path)
    return out


def serve(args):
    import uvicorn

    from .api_server import ServerSettings, create_app

    settings = ServerSettings(args.model, args.served_model_name, args.api_key or os.environ.get("DAN_API_KEY", ""),
                              load_profiles(args.profile))
    uvicorn.run(create_app(settings, build_engine(args)), host=args.host, port=args.port)


def make_llm(args):
    from .llm import LLM

    return LLM(args.model, args.device, DTYPES[args.dtype], adapter=args.adapter, quantization=args.quantization)


def fmt(m):
    return f"acc {m['accuracy']:.3f}  ece {m['ece']:.3f}  brier {m['brier']:.3f}  nll {m['nll']:.3f}  (n={m['n']})"


def run_eval(args):
    from ..calibration import Profile
    from ..tuning import evaluate, load_rows

    profile = Profile.load(args.profile) if args.profile else None
    if profile is None and (args.permutations > 1 or args.label_style != "letters" or args.layout != "inline"):
        profile = Profile(args.permutations, args.label_style, args.layout)
    report = evaluate(make_llm(args), load_rows(args.data)[: args.limit or None], profile)
    for key, m in report.items():
        print(f"{key:16s} {fmt(m)}")
    if args.json:
        print(json.dumps(report))


def run_calibrate(args):
    from ..tuning import calibrate, load_rows

    eval_rows = load_rows(args.eval)[: args.limit or None] if args.eval else None
    profile, report = calibrate(make_llm(args), load_rows(args.data)[: args.limit or None], eval_rows,
                                args.method, args.permutations, args.label_style, layout=args.layout)
    profile.save(args.out)
    print(f"wrote {args.out}")
    for key, r in report.items():
        print(f"{key:16s} raw        {fmt(r['raw'])}")
        print(f"{'':16s} calibrated {fmt(r['calibrated'])}")
    if args.json:
        print(json.dumps(report))


def run_train(args):
    from transformers import AutoTokenizer

    from .. import lora
    from ..prompt import Planner
    from ..runner import Runner
    from ..tuning import load_rows, train_lora

    runner = Runner(args.model, args.device, DTYPES[args.dtype], cache_bytes=0)
    planner = Planner(AutoTokenizer.from_pretrained(args.model))
    train_lora(runner, planner, load_rows(args.data)[: args.limit or None], args.epochs, args.lr, args.rank,
               args.alpha, args.batch_size, args.loss, args.label_style, layout=args.layout)
    lora.save(runner.model, args.out)
    print(f"wrote {args.out}")


def model_args(p, adapter=True):
    p.add_argument("model", help="Hugging Face model id or local path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=DTYPES, default="bfloat16" if torch.cuda.is_available() else "float32")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = default)")
    if adapter:
        p.add_argument("--adapter", default=None, help="LoRA adapter directory from `dan train`")
        p.add_argument("--quantization", choices=("fp8",), default=None, help="FP8 weights+activations (sm89+)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dan")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve a model over the Jev-compatible HTTP API")
    model_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--served-model-name", default="dan-latest")
    s.add_argument("--profile", action="append", help="NAME=profile.json; request it as model '<served>@NAME'")
    s.add_argument("--api-key", default="", help="require this bearer token (or set DAN_API_KEY)")
    s.add_argument("--max-batch-tokens", type=int, default=8192)
    s.add_argument("--max-wait-ms", type=float, default=2.0)
    s.add_argument("--max-queue", type=int, default=1024)
    s.add_argument("--prefix-cache-gb", type=float, default=2.0)
    s.set_defaults(func=serve)

    for name, func, text in (("eval", run_eval, "measure accuracy and calibration on labeled JSONL"),
                             ("calibrate", run_calibrate, "fit a calibration profile on labeled JSONL")):
        c = sub.add_parser(name, help=text)
        model_args(c)
        c.add_argument("data", help="labeled JSONL")
        c.add_argument("--permutations", type=int, default=1, help="option rotations read and averaged")
        c.add_argument("--label-style", choices=("letters", "names"), default="letters")
        c.add_argument("--layout", choices=("inline", "system"), default="inline")
        c.add_argument("--limit", type=int, default=0)
        c.add_argument("--json", action="store_true", help="also print the report as JSON")
        if name == "eval":
            c.add_argument("--profile", default=None, help="profile.json to apply")
        else:
            c.add_argument("--eval", default=None, help="held-out labeled JSONL to report on")
            c.add_argument("--method", choices=("temperature", "vector", "contextual"), default="vector")
            c.add_argument("--out", required=True)
        c.set_defaults(func=func)

    t = sub.add_parser("train", help="LoRA fine-tune for calibrated decisions on labeled JSONL")
    model_args(t, adapter=False)
    t.add_argument("data")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--loss", choices=("log", "brier"), default="log")
    t.add_argument("--label-style", choices=("letters", "names"), default="letters")
    t.add_argument("--layout", choices=("inline", "system"), default="inline")
    t.add_argument("--limit", type=int, default=0)
    t.set_defaults(func=run_train)

    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
