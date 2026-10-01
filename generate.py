"""Generate text from a trained checkpoint.

Usage:
  .venv/bin/python generate.py --ckpt out/tinystories/ckpt.pt "Once upon a time"
"""

import argparse
import sys
import time

import torch

from codec import Codec
from model import LLM, Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=args.device)
    model = LLM(Config(**ck["config"])).to(args.device).eval()
    model.load_state_dict(ck["model"])
    codec = Codec(ck["data"])
    print(f"[iter {ck['iter']}, val loss {ck['best_val']:.3f}, {model.num_params() / 1e6:.1f}M params]\n")

    ids = codec.encode(args.prompt) if args.prompt else [codec.eos_id if codec.eos_id is not None else 0]
    for _ in range(args.num_samples):
        idx = torch.tensor([ids], device=args.device)
        print(args.prompt, end="", flush=True)
        out, printed = [], ""
        t0 = time.time()
        for t in model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, eos_id=codec.eos_id):
            if t == codec.eos_id:
                break
            out.append(t)
            text = codec.decode(out)  # decode everything and print the diff so multi-byte chars are never split
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
        dt = time.time() - t0
        print(f"\n\n[{len(out)} tokens, {len(out) / dt:.1f} tok/s on {args.device}]")
        print("-" * 40)


if __name__ == "__main__":
    main()
