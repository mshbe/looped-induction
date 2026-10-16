"""Train a looped attention-only transformer on TinyStories (roneneldan/TinyStories).

Run from the project folder, e.g.:
    uv run python training.py --max_steps 30 --n_train_stories 5000 --run_name smoke   # quick test
    uv run python training.py --n_layers 1 --n_loops 2 --seed 0   # looped model
    uv run python training.py --n_layers 1 --n_loops 1 --seed 0   # 1-layer control
    uv run python training.py --n_layers 2 --n_loops 1 --seed 0 --device=cpu   # 2-layer control

    uv run python training.py --max_steps 20000 --n_layers 1 --n_loops 2 --seed 0 --run_name ts_L1_loops2_seed0_20k --ckpt_every 1000     # extensive test

Default run names start with ts_ (TinyStories), so they don't overwrite the earlier pile-10k runs.

Sanity check with synthetic data (repeated random segments, see make_synthetic_batch):
    uv run python training.py --data synthetic --n_layers 2 --n_loops 1 --max_steps 3000 --batch_size 32
Run names then start with syn_. val_loss is still measured on TinyStories, so it stays high; watch
the repeated-token numbers instead.
The first run downloads and tokenizes the data (a few minutes); later runs reuse the files in data/.

Each run writes to checkpoints/<run_name>/:
    config.json      the model Config
    metrics.jsonl    one line of numbers per evaluation
    step_XXXXXX.pt   saved weights at intervals
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch as t
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformer_lens.utils import tokenize_and_concatenate
from transformers import AutoTokenizer

from model import Config, LoopedAttnOnly

DATASET = "roneneldan/TinyStories"
DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")


# ---------------------------------------------------------------- setup helpers

def get_device(name: str | None) -> t.device:
    if name is not None:
        return t.device(name)
    if t.backends.mps.is_available():
        return t.device("mps")
    if t.cuda.is_available():
        return t.device("cuda")
    return t.device("cpu")


def _tokenize(dataset, tokenizer, n_ctx: int) -> t.Tensor:
    """Join the stories end to end (separated by an end-of-text token) and cut the result into
    rows of n_ctx tokens, each starting with BOS. A row can span the end of one story and the
    start of the next. Stored as int32 to halve the file size; convert with .long() before use.
    """
    tokenized = tokenize_and_concatenate(
        dataset, tokenizer, max_length=n_ctx, add_bos_token=True, num_proc=4
    )
    return tokenized.with_format("torch")[:]["tokens"].to(t.int32)


def load_train_tokens(tokenizer, n_ctx: int, n_stories: int) -> t.Tensor:
    """The first n_stories TinyStories training stories, as a [n_rows, n_ctx] int32 tensor.

    Streaming downloads only the stories needed, not all 2.1M. The result is saved to data/
    and reused on later runs with the same n_stories and n_ctx.
    """
    path = DATA_DIR / f"tinystories_train{n_stories}_ctx{n_ctx}.pt"
    if path.exists():
        return t.load(path)
    stream = load_dataset(DATASET, split="train", streaming=True)
    dataset = Dataset.from_list(list(stream.take(n_stories)))
    tokens = _tokenize(dataset, tokenizer, n_ctx)
    DATA_DIR.mkdir(exist_ok=True)
    t.save(tokens, path)
    return tokens


def load_val_tokens(tokenizer, n_ctx: int) -> t.Tensor:
    """TinyStories' separate validation split (~22k stories), same format as load_train_tokens."""
    path = DATA_DIR / f"tinystories_val_ctx{n_ctx}.pt"
    if path.exists():
        return t.load(path)
    dataset = load_dataset(DATASET, split="validation")
    tokens = _tokenize(dataset, tokenizer, n_ctx)
    DATA_DIR.mkdir(exist_ok=True)
    t.save(tokens, path)
    return tokens


def common_token_ids(tokens: t.Tensor, d_vocab: int, k: int, exclude: list[int]) -> t.Tensor:
    """The k most frequent token ids in `tokens`, leaving out `exclude` (special tokens like BOS).

    The repeated-token test draws its random tokens from these. TinyStories uses only a small
    part of the tokenizer's ~50k-token vocabulary.
    """
    flat = tokens.flatten().long()
    counts = t.bincount(flat, minlength=max(d_vocab, int(flat.max()) + 1))[:d_vocab]
    for i in exclude:
        if i < d_vocab:
            counts[i] = 0
    return counts.topk(k).indices.sort().values


def lr_at(step: int, args) -> float:
    """Linear warmup, then cosine decay down to 10% of the peak learning rate."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------- losses / evals

def per_token_loss(logits: t.Tensor, tokens: t.Tensor) -> t.Tensor:
    """Next-token cross-entropy at each position. Output shape [batch, seq-1]."""
    batch, seq = tokens.shape
    return F.cross_entropy(
        logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten(), reduction="none"
    ).view(batch, seq - 1)


def make_repeated_tokens(bos_id: int, token_ids: t.Tensor, half_len: int, batch: int) -> t.Tensor:
    """[BOS, r_1..r_L, r_1..r_L] with each r_i drawn at random from token_ids (see common_token_ids).
    Fixed seed, so every evaluation uses the same batch."""
    g = t.Generator().manual_seed(1234)
    rand = token_ids[t.randint(0, len(token_ids), (batch, half_len), generator=g)]
    bos = t.full((batch, 1), bos_id)
    return t.cat([bos, rand, rand], dim=1)


def make_synthetic_batch(bos_id: int, token_ids: t.Tensor, n_ctx: int, batch: int,
                         gen: t.Generator, min_len: int = 8) -> t.Tensor:
    """Synthetic training data: each row is BOS, then a random segment repeated until the row is full.

    The segment length L is drawn per row from [min_len, (n_ctx-1)//2], so a model can't solve the
    task by always attending a fixed distance back (a 1-layer model could do that). Copying has to
    be content-based: find the earlier occurrence of the current token, predict what followed it.
    That is what an induction head does, and a 1-layer model shouldn't manage it.
    """
    max_len = (n_ctx - 1) // 2
    rows = []
    for _ in range(batch):
        L = int(t.randint(min_len, max_len + 1, (1,), generator=gen))
        seg = token_ids[t.randint(0, len(token_ids), (L,), generator=gen)]
        n_rep = -(-(n_ctx - 1) // L)   # ceiling division
        rows.append(seg.repeat(n_rep)[: n_ctx - 1])
    return t.cat([t.full((batch, 1), bos_id), t.stack(rows)], dim=1)


@t.no_grad()
def evaluate(model, val_tokens, rep_tokens, half_len, args, device) -> dict:
    model.eval()

    # held-out text loss
    losses = []
    for i in range(args.val_batches):
        batch = val_tokens[i * args.batch_size : (i + 1) * args.batch_size].to(device).long()
        losses.append(per_token_loss(model(batch), batch).mean().item())

    # repeated random tokens: first copy is unpredictable (loss at least ~ ln rep_vocab_size);
    # the second copy can be predicted by induction. Loss index i predicts token i+1.
    # Index half_len predicts r_1 from r_L, which induction can't do, so the second half starts at half_len+1.
    rep = rep_tokens.to(device)
    rep_loss = per_token_loss(model(rep), rep)
    first = rep_loss[:, :half_len].mean().item()
    second = rep_loss[:, half_len + 1 :].mean().item()

    model.train()
    return {"val_loss": sum(losses) / len(losses), "rep_first_half": first, "rep_second_half": second}


def save_checkpoint(model, run_dir: Path, step: int):
    t.save({"step": step, "state_dict": model.state_dict()}, run_dir / f"step_{step:06d}.pt")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_loops", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--val_batches", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--device", default=None, help="mps, cpu, or cuda (default: best available)")
    p.add_argument("--run_name", default=None)
    p.add_argument("--data", choices=["tinystories", "synthetic"], default="tinystories",
                   help="synthetic = repeated random segments, a sanity check that induction can form")
    p.add_argument("--n_train_stories", type=int, default=300_000,
                   help="how many TinyStories training stories to use (~2.1M available)")
    p.add_argument("--rep_vocab_size", type=int, default=2000,
                   help="repeated-token test draws from this many of the most common tokens")
    args = p.parse_args()

    prefix = "ts" if args.data == "tinystories" else "syn"
    run_name = args.run_name or f"{prefix}_L{args.n_layers}_loops{args.n_loops}_seed{args.seed}"
    run_dir = CKPT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    # tokenizer and config. d_vocab is read before loading the data, which may add a padding
    # token to the tokenizer; padding never appears in the training data.
    cfg = Config(n_layers=args.n_layers, n_loops=args.n_loops)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    cfg.d_vocab = len(tokenizer)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    # data: TinyStories has a separate validation split, so nothing is held out of train
    val_tokens = load_val_tokens(tokenizer, cfg.n_ctx)
    half_len = (cfg.n_ctx - 1) // 2
    rep_ids = common_token_ids(val_tokens, cfg.d_vocab, args.rep_vocab_size, tokenizer.all_special_ids)
    rep_tokens = make_repeated_tokens(tokenizer.bos_token_id, rep_ids, half_len, args.batch_size)
    if args.data == "tinystories":
        train_tokens = load_train_tokens(tokenizer, cfg.n_ctx, args.n_train_stories)
        print(f"{len(train_tokens)} training rows of {cfg.n_ctx} tokens "
              f"(~{len(train_tokens) // args.batch_size} steps per pass through the data)")
    else:
        print(f"synthetic data: repeated random segments of {len(rep_ids)} common tokens")

    # model and optimizer
    t.manual_seed(args.seed)
    model = LoopedAttnOnly(cfg).to(device)
    n_embed = model.embed.W_E.numel() + model.unembed.W_U.numel() + model.pos_embed.W_pos.numel()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params: {n_total:,} total, {n_total - n_embed:,} in attention")
    optimizer = t.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data_gen = t.Generator().manual_seed(args.seed)

    metrics_file = open(run_dir / "metrics.jsonl", "w")
    ema = None
    start = time.time()
    for step in range(args.max_steps + 1):
        if step % args.eval_every == 0 or step == args.max_steps:
            m = evaluate(model, val_tokens, rep_tokens, half_len, args, device)
            m.update(step=step, train_loss_ema=ema, lr=lr_at(step, args), minutes=(time.time() - start) / 60)
            metrics_file.write(json.dumps(m) + "\n")
            metrics_file.flush()
            print(f"step {step:5d} | train {ema or float('nan'):.3f} | val {m['val_loss']:.3f} | "
                  f"repeated tokens: 1st half {m['rep_first_half']:.2f}, 2nd half {m['rep_second_half']:.2f} | "
                  f"{m['minutes']:.1f} min")
        if step % args.ckpt_every == 0 or step == args.max_steps:
            save_checkpoint(model, run_dir, step)
        if step == args.max_steps:
            break

        # one training step on a random batch (sampled with replacement, or generated fresh)
        if args.data == "tinystories":
            idx = t.randint(0, len(train_tokens), (args.batch_size,), generator=data_gen)
            batch = train_tokens[idx].to(device).long()
        else:
            batch = make_synthetic_batch(tokenizer.bos_token_id, rep_ids, cfg.n_ctx,
                                         args.batch_size, data_gen).to(device)
        loss = per_token_loss(model(batch), batch).mean()

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        optimizer.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()

    metrics_file.close()
    print(f"done: {run_dir}")


# The guard is required on macOS: tokenizing uses several processes, and each one re-imports this file.
if __name__ == "__main__":
    main()
