"""TinyStoriesV2 (GPT-4 generated): train a byte-level BPE tokenizer and encode the dataset.

Downloads data/TinyStoriesV2-GPT4-{train,valid}.txt (2.2 GB) from the
roneneldan/TinyStories dataset on the Hugging Face Hub if missing.
Writes data/tinystories/{tokenizer.json, train.bin, val.bin, meta.json}.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"

ap = argparse.ArgumentParser()
ap.add_argument("--vocab-size", type=int, default=4096)
ap.add_argument("--tokenizer-sample-mb", type=int, default=200, help="how much text to train the tokenizer on")
args = ap.parse_args()

src = Path("data")
out = Path("data/tinystories")
out.mkdir(parents=True, exist_ok=True)
for name in ("valid", "train"):
    fname = f"TinyStoriesV2-GPT4-{name}.txt"
    if not (src / fname).exists():
        print(f"downloading {fname}")
        hf_hub_download("roneneldan/TinyStories", fname, repo_type="dataset", local_dir=src)


def stories(path):
    """The files separate stories with <|endoftext|>."""
    for s in path.read_text(encoding="utf-8").split(EOS):
        s = s.strip()
        if s:
            yield s


# 1. Train the tokenizer: start from the 256 byte values and learn merges up to vocab_size.
t0 = time.time()
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=[EOS],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
with open(src / "TinyStoriesV2-GPT4-train.txt", encoding="utf-8") as f:
    sample = f.read(args.tokenizer_sample_mb * 1_000_000)
tok.train_from_iterator([s for s in sample.split(EOS) if s.strip()], trainer)
tok.save(str(out / "tokenizer.json"))
eos_id = tok.token_to_id(EOS)
print(f"tokenizer: vocab {tok.get_vocab_size()} trained on {args.tokenizer_sample_mb}MB in {time.time() - t0:.0f}s")

# 2. Encode every story and append <|endoftext|> so the model learns where stories end.
for split, name in (("val", "valid"), ("train", "train")):
    t0 = time.time()
    all_stories = list(stories(src / f"TinyStoriesV2-GPT4-{name}.txt"))
    with open(out / f"{split}.bin", "wb") as f:
        n_tokens = 0
        for i in range(0, len(all_stories), 20_000):
            batch = tok.encode_batch(all_stories[i:i + 20_000])
            ids = np.fromiter((t for e in batch for t in (*e.ids, eos_id)), dtype=np.uint16)
            ids.tofile(f)
            n_tokens += len(ids)
    print(f"{split}: {len(all_stories):,} stories, {n_tokens:,} tokens in {time.time() - t0:.0f}s")

(out / "meta.json").write_text(json.dumps({"type": "bpe", "eos_token": EOS}))
