"""Train a looped attention-only transformer on text (NeelNanda/pile-10k).

Run from the project folder, e.g.:
    uv run python training.py --max_steps 30 --run_name smoke     # quick test
    uv run python training.py --n_layers 1 --n_loops 2 --seed 0   # looped model
    uv run python training.py --n_layers 1 --n_loops 1 --seed 0   # 1-layer control
    uv run python training.py --n_layers 2 --n_loops 1 --seed 0 --device=cpu   # 2-layer control

    uv run python training.py --max_steps 20000 --n_layers 1 --n_loops 2 --seed 0 --run_name L1_loops2_seed0_20k --ckpt_every 1000     # extensive test

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
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
from transformers import AutoTokenizer

from model import Config, LoopedAttnOnly

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


def load_tokens(tokenizer, n_ctx: int) -> t.Tensor:
    """Return pile-10k as a [n_rows, n_ctx] tensor of token ids, each row starting with BOS.

    Tokenizing takes a few minutes, so the result is saved to data/ and reused on later runs.
    """
    path = DATA_DIR / f"pile10k_ctx{n_ctx}.pt"
    if path.exists():
        return t.load(path)
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    tokenized = tokenize_and_concatenate(
        dataset, tokenizer, max_length=n_ctx, add_bos_token=True, num_proc=4
    )
    tokens = tokenized.with_format("torch")[:]["tokens"]
    DATA_DIR.mkdir(exist_ok=True)
    t.save(tokens, path)
    return tokens


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


def make_repeated_tokens(bos_id: int, d_vocab: int, half_len: int, batch: int) -> t.Tensor:
    """[BOS, r_1..r_L, r_1..r_L] with random r_i. Fixed seed, so every evaluation uses the same batch."""
    g = t.Generator().manual_seed(1234)
    rand = t.randint(0, d_vocab, (batch, half_len), generator=g)
    bos = t.full((batch, 1), bos_id)
    return t.cat([bos, rand, rand], dim=1)


@t.no_grad()
def evaluate(model, val_tokens, rep_tokens, half_len, args, device) -> dict:
    model.eval()

    # held-out text loss
    losses = []
    for i in range(args.val_batches):
        batch = val_tokens[i * args.batch_size : (i + 1) * args.batch_size].to(device)
        losses.append(per_token_loss(model(batch), batch).mean().item())

    # repeated random tokens: first copy is unpredictable (loss ~ ln d_vocab);
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
    args = p.parse_args()

    run_name = args.run_name or f"L{args.n_layers}_loops{args.n_loops}_seed{args.seed}"
    run_dir = CKPT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    # tokenizer and config. d_vocab is read before load_tokens, which may add a padding
    # token to the tokenizer; padding never appears in the training data.
    cfg = Config(n_layers=args.n_layers, n_loops=args.n_loops)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    cfg.d_vocab = len(tokenizer)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    # data: hold out the last 2% of rows for validation
    tokens = load_tokens(tokenizer, cfg.n_ctx)
    n_val = max(args.val_batches * args.batch_size, len(tokens) // 50)
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]
    half_len = (cfg.n_ctx - 1) // 2
    rep_tokens = make_repeated_tokens(tokenizer.bos_token_id, cfg.d_vocab, half_len, args.batch_size)
    print(f"{len(train_tokens)} training rows of {cfg.n_ctx} tokens "
          f"(~{len(train_tokens) // args.batch_size} steps per pass through the data)")

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

        # one training step on a random batch (sampled with replacement)
        idx = t.randint(0, len(train_tokens), (args.batch_size,), generator=data_gen)
        batch = train_tokens[idx].to(device)
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
