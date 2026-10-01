"""Character-level Tiny Shakespeare: every distinct character is one token.

Downloads data/shakespeare.txt (1.1 MB) if missing.
Writes data/shakespeare/{train.bin, val.bin, meta.json}.
"""

import json
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

src = Path("data/shakespeare.txt")
out = Path("data/shakespeare")
out.mkdir(parents=True, exist_ok=True)
if not src.exists():
    print(f"downloading {URL}")
    ctx = ssl.create_default_context(cafile=certifi.where())  # python.org builds on macOS ship without CA certs
    with urllib.request.urlopen(URL, context=ctx) as r:
        src.write_bytes(r.read())

text = src.read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)

n = int(len(ids) * 0.9)
ids[:n].tofile(out / "train.bin")
ids[n:].tofile(out / "val.bin")
(out / "meta.json").write_text(json.dumps({"type": "char", "chars": chars}))
print(f"{len(text):,} chars, vocab {len(chars)}, train {n:,} / val {len(ids) - n:,} tokens")
