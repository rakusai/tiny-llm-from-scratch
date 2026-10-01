"""Text <-> token ID conversion for a prepared dataset directory (char-level or BPE)."""

import json
from pathlib import Path


class Codec:
    def __init__(self, data_dir):
        meta = json.loads((Path(data_dir) / "meta.json").read_text())
        self.type = meta["type"]
        if self.type == "char":
            self.chars = meta["chars"]
            self.stoi = {c: i for i, c in enumerate(self.chars)}
            self.vocab_size = len(self.chars)
            self.eos_id = None
        else:
            from tokenizers import Tokenizer

            self.tok = Tokenizer.from_file(str(Path(data_dir) / "tokenizer.json"))
            self.vocab_size = self.tok.get_vocab_size()
            self.eos_id = self.tok.token_to_id(meta["eos_token"])

    def encode(self, text: str) -> list[int]:
        if self.type == "char":
            return [self.stoi[c] for c in text]
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        if self.type == "char":
            return "".join(self.chars[i] for i in ids)
        return self.tok.decode(ids)
