# tiny-llm-from-scratch

Train a small Llama-style language model from scratch on a laptop. On an Apple M4 MacBook, **three minutes of training** turns random weights into a model that writes short children's stories:

> Once upon a time, there was a funny bear. He was very helpful to help his friends.
> One day, a little boy named Tim came to the park. He saw a big, round ball on the ground. … They played all day long. They laughed and played with the ball, and the bird was very happy.

*(1.3M parameters, 3 minutes on the M4 GPU, trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories))*

Everything here is written in plain PyTorch: the model (RMSNorm, grouped-query attention, RoPE, SwiGLU), the training loop, data preparation, and a byte-level BPE tokenizer trained on the dataset itself.

## Try it: a story-writing model in 3 minutes

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# One-time setup (~15 min): download TinyStories (2.2 GB), train a 4,096-token BPE tokenizer, encode the text
.venv/bin/python prepare_tinystories.py

# Train a 3.15M-parameter model for exactly 3 minutes
.venv/bin/python train.py --data data/tinystories --out out/ts-3min --time-budget 180

# Write a story
.venv/bin/python generate.py --ckpt out/ts-3min/ckpt.pt "Once upon a time"
```

`--time-budget` sets the training time in seconds and fits the learning-rate schedule to it, so the run finishes at a low learning rate instead of being cut off mid-schedule. Evaluation time is not counted.

Every few hundred steps the training script prints the loss and a sample story, so you can watch the text go from random words to grammatical sentences.

## Results on an Apple M4

All runs on TinyStories, batch 32 × 256 tokens, M4 GPU (MPS):

| Model | Training time | Tokens read | Validation loss |
|---|---|---|---|
| 1.31M params | 3 min | 6.8M | 2.495 |
| 3.15M params | 3 min | 4.4M | 2.503 |
| 3.15M params | 10 min | 12.1M | **2.052** |

What the 3.15M model writes after 10 minutes:

> Once upon a time, there was a little boy named Tim. Tim had a toy box. He loved to play with his toy. … One day, Tim saw his friend, Sam the bird. Sam was not nice. He wanted to play with Tim. … Tim felt bad for not listening to Sam. He said, "I will be friends again soon." Sam and Tim played outside together. They had so much fun.

Some takeaways:

- **Three minutes is enough for grammar.** Sentences and dialogue are mostly correct; the plot still wanders.
- **Ten minutes brings story structure.** A typical arc appears (a falling-out, regret, making up), though the model still mixes up who did what.
- **At a 3-minute budget, model size barely matters.** The 1.31M model runs nearly twice as fast and reads more data, which makes up for being smaller.
- **Finish the learning-rate schedule.** Stopping a longer run at the same point gave a loss of 2.571, versus 2.503 when the schedule was fitted to the time.

## How it works

An illustrated walkthrough of tokenization, the training step, loss, the learning-rate schedule, and the trade-off between model size and training time, with charts from these runs:

- [How Training Works](docs/how-training-works.html) (English)
- [TinyStories 学習図解](docs/how-training-works-ja.html) (Japanese)

## Warm-up: Shakespeare in 8 minutes

A character-level model is a quick way to check that everything runs. The data is 1.1 MB and downloads automatically.

```bash
.venv/bin/python prepare_shakespeare.py
.venv/bin/python train.py --data data/shakespeare --out out/shakespeare \
    --layers 4 --dim 128 --heads 4 --kv-heads 4 --ffn 384 --max-iters 2000
.venv/bin/python generate.py --ckpt out/shakespeare/ckpt.pt "ROMEO:"
```

## Files

| File | Purpose |
|---|---|
| `model.py` | The model: embedding, decoder layers (RMSNorm, GQA with RoPE, SwiGLU), KV-cached generation |
| `train.py` | Training loop: random batches, AdamW, warmup + cosine schedule, evaluation, checkpoints |
| `generate.py` | Text generation from a checkpoint, with tokens/s |
| `codec.py` | Text ↔ token IDs for character-level or BPE datasets |
| `prepare_tinystories.py` | Downloads TinyStories, trains the BPE tokenizer, writes `train.bin` / `val.bin` |
| `prepare_shakespeare.py` | Downloads Tiny Shakespeare and writes character-level `train.bin` / `val.bin` |
| `docs/` | Illustrated explanation (English and Japanese) |

Each training run writes to its own `out/<run name>/` folder: `ckpt.pt` (weights, optimizer state and model shape), `log.jsonl` (loss at each evaluation) and `train.log` (full console output). Add `--resume` to continue a run.

## Main options

| Option | Default | Meaning |
|---|---|---|
| `--time-budget` | off | Train for this many seconds (overrides `--max-iters`) |
| `--max-iters` | 5000 | Number of training steps |
| `--dim` / `--layers` | 192 / 6 | Hidden size and number of layers |
| `--heads` / `--kv-heads` | 6 / 2 | Query heads and key/value heads (grouped-query attention) |
| `--ffn` | 512 | MLP hidden size |
| `--batch-size` / `--seq-len` | 32 / 256 | Sequences per step and tokens per sequence |
| `--lr` / `--min-lr` / `--warmup` | 2e-3 / 2e-4 / 100 | Peak and final learning rate, warmup steps |
| `--device` | `mps` if available | `mps`, `cuda` or `cpu` |

The defaults are the 3.15M configuration used above. On the M4, float32 trained faster than bfloat16, so it is the default.

## Related

[smollm2-from-scratch](https://github.com/rakusai/smollm2-from-scratch): the same architecture at 135M parameters, loading the official SmolLM2 weights instead of training.

## License

MIT for the code in this repository. The datasets are downloaded from their original sources and are not included.
