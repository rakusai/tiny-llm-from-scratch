"""Pretrain the model from scratch on a prepared dataset.

Usage:
  # 3.15M-parameter model on TinyStories for 3 minutes
  .venv/bin/python train.py --data data/tinystories --out out/ts-3min --time-budget 180

  # tiny character-level model on Shakespeare
  .venv/bin/python train.py --data data/shakespeare --out out/shakespeare \
      --layers 4 --dim 128 --heads 4 --kv-heads 4 --ffn 384 --max-iters 2000
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from codec import Codec
from model import LLM, Config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with train.bin, val.bin, meta.json")
    ap.add_argument("--out", required=True, help="directory for checkpoints and logs")
    # model shape
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=256)
    # optimization
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--time-budget", type=float, default=0,
                    help="train for this many seconds instead of --max-iters (eval time excluded)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # bookkeeping
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--sample-prompt", default="\n")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])  # fp32 was faster on M4
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


class Tee:
    """Write everything printed to the terminal into a log file as well."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()


def lr_at(it, progress, args):
    """Linear warmup, then cosine decay down to min_lr as progress goes from 0 to 1."""
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = min(1.0, progress)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    dev = args.device

    codec = Codec(args.data)
    data = {s: np.memmap(Path(args.data) / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}

    def get_batch(split):
        d = data[split]
        ix = np.random.randint(0, len(d) - args.seq_len - 1, args.batch_size)
        x = torch.from_numpy(np.stack([d[i:i + args.seq_len] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([d[i + 1:i + 1 + args.seq_len] for i in ix]).astype(np.int64))
        return x.to(dev), y.to(dev)

    cfg = Config(
        vocab_size=codec.vocab_size,
        hidden_size=args.dim,
        intermediate_size=args.ffn,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.seq_len,
    )
    model = LLM(cfg).to(dev)

    # weight decay only on matrices, not on norms
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    start_iter, best_val = 0, float("inf")
    if args.resume and (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_iter, best_val = ck["iter"] + 1, ck["best_val"]
        print(f"resumed from iter {ck['iter']}")

    tokens_per_iter = args.batch_size * args.seq_len * args.grad_accum
    budget = f"{args.time_budget:.0f}s" if args.time_budget else \
        f"{args.max_iters * tokens_per_iter / 1e6:.0f}M tokens total"
    print(f"params {model.num_params() / 1e6:.2f}M | vocab {cfg.vocab_size} | "
          f"{tokens_per_iter:,} tokens/iter | {budget} | {dev}")
    autocast = torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=args.dtype == "bfloat16")

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(args.eval_iters):
                x, y = get_batch(split)
                with autocast:
                    losses.append(model(x, targets=y)[1].item())
            res[split] = sum(losses) / len(losses)
        model.train()
        return res

    def sample(n=120):
        model.eval()
        idx = torch.tensor([codec.encode(args.sample_prompt)], device=dev)
        text = codec.decode(list(model.generate(idx, n, temperature=0.8, top_k=40, eos_id=codec.eos_id)))
        model.train()
        return args.sample_prompt + text

    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    train_start, eval_time = time.time(), 0.0
    it = start_iter
    while True:
        if args.time_budget:
            progress = (time.time() - train_start - eval_time) / args.time_budget
        else:
            progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        done = progress >= 1 if args.time_budget else it >= args.max_iters
        lr = lr_at(it, progress, args)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or done:
            t_eval = time.time()
            ev = evaluate()
            print(f"--- iter {it}: train {ev['train']:.4f} | val {ev['val']:.4f}")
            print(sample().strip()[:400])
            print("---")
            log.write(json.dumps({"iter": it, "train": ev["train"], "val": ev["val"]}) + "\n")
            log.flush()
            if ev["val"] < best_val:
                best_val = ev["val"]
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg.to_dict(),
                        "data": str(args.data), "iter": it, "best_val": best_val}, out / "ckpt.pt")
            eval_time += time.time() - t_eval
            t0 += time.time() - t_eval  # keep eval out of the tok/s figure
        if done:
            print(f"done: {it} iters, {it * tokens_per_iter / 1e6:.1f}M tokens, "
                  f"{time.time() - train_start - eval_time:.0f}s training")
            break

        for _ in range(args.grad_accum):
            x, y = get_batch("train")
            with autocast:
                _, loss = model(x, targets=y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if it % args.log_interval == 0:
            loss_val = loss.item()  # forces a device sync, so timing below is real
            dt = time.time() - t0
            t0 = time.time()
            n = args.log_interval if it > start_iter else 1
            print(f"iter {it:6d} | loss {loss_val:.4f} | lr {lr:.2e} | "
                  f"{n * tokens_per_iter / dt:,.0f} tok/s")
        it += 1


if __name__ == "__main__":
    main()
