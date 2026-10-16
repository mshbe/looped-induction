"""Replicate Aoyama et al.'s 2-layer setup, where induction heads are known to form, then loop it.

Paper: "Predicting the Emergence of Induction Heads in Language Model Pretraining" (arXiv 2511.16893),
code: github.com/t-aoyam/predict-ih-review. What is copied from their code:
  - model: Hugging Face GPT-2 from scratch, 2 layers, 8 heads, d_model 768, MLP, pre-LayerNorm,
    learned absolute positions, dropout 0.1, tied embedding/unembedding
  - data: CC100 English, GPT-2 tokenizer. Each line is tokenized separately and the pieces are
    joined with no separator; a blank line (end of document) becomes an end-of-text token. The
    token stream is cut into rows of ctx tokens. End-of-text targets are left out of the loss
    (their data collator treats end-of-text as padding).
  - optimizer: Hugging Face Trainer defaults, i.e. AdamW (0.9, 0.999), eps 1e-8, weight decay 0,
    gradient clipping at 1.0; lr 2.5e-4 (their config file; the paper's table says 5e-4);
    cosine schedule with warmup over the first 1% of steps
  - the schedule is set for 1B tokens even though we stop much earlier (--max_tokens), so the
    learning rate stays near its peak over the window where induction heads should form
  - induction score (their "prefix-matching score"): 50 distinct random tokens, repeated twice, no
    BOS; a head's score is its mean attention from each second-copy token to the token after that
    token's first occurrence. 100 such sequences, fixed seed.

Their fitted law predicts formation at about e^13.26 * B^-0.37 * C^-0.62 steps (B = batch size,
C = context): about 2,800 steps (46M tokens) for the default B=16, C=1024.

Looping: --n_layers L --n_loops K builds L*K blocks and makes block i the same module as block
i % L, so the weights are shared, not copied. The tie is checked after every step and at every
checkpoint.

Stepping down towards the attention-only model in model.py (apply to both the control and the twin):
  --no_mlp        remove the MLPs
  --no_ln         remove every LayerNorm
  --dropout 0     no dropout (model.py has none)
  --arch mine     model.py's LoopedAttnOnly: shortformer positions, no biases, untied unembedding

Run from the project folder:
    uv run python replicate.py --prepare_only                         # download + tokenize (~110M tokens)
    uv run python replicate.py --n_layers 2 --n_loops 1               # step 1: replication
    uv run python replicate.py --n_layers 1 --n_loops 2               # looped twin
    uv run python replicate.py --n_layers 2 --n_loops 1 --no_mlp      # and so on down the ladder
    uv run python replicate.py --bf16 ...                             # faster on a CUDA GPU
    uv run python replicate.py --ckpt_phases 0.03,0.1,0.25 ...        # save only 4 checkpoints (below)
Quick CPU test:
    uv run python replicate.py --data_tokens 3000000 --d_model 64 --ctx 128 --batch_size 8 \
        --max_tokens 200000 --eval_every 50 --device cpu --run_name smoke_rep

Each run writes to checkpoints/<run_name>/:
    train_args.json  the arguments (load_run rebuilds the model from these)
    metrics.jsonl    one line per evaluation, including every head's induction and previous-token score
    step_XXXXXX.pt   saved weights at the --ckpt_mtokens milestones and at the end
With --ckpt_phases PRE,ONSET,MID, the checkpoints are chosen by the induction score instead:
pre-formation (the last of the saves made every --latest_mtokens before the score first exceeds
PRE), onset (first evaluation above ONSET), mid-transition (first above MID) and the end.
phases.json maps each of these labels to its step; latest.pt is the pre-formation candidate
while the run is still before the crossing.
"""

import argparse
import json
import lzma
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch as t
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from model import Config, LoopedAttnOnly

CC100_URL = "https://data.statmt.org/cc-100/en.txt.xz"
DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
VOCAB = 50257   # GPT-2 tokenizer
EOS = 50256     # <|endoftext|>
VAL_TOKENS = 1_000_000   # the last this-many prepared tokens are held out for validation


# ---------------------------------------------------------------- data

def prepare_cc100(n_tokens: int) -> Path:
    """Stream the start of CC100 English and tokenize it the way Aoyama et al. do (see docstring).

    Saved as a uint16 .npy file and reused by later runs asking for the same number of tokens.
    Only the first ~0.5GB of the 87GB compressed file is downloaded.
    """
    path = DATA_DIR / f"cc100_gpt2_{n_tokens}.npy"
    if path.exists():
        return path
    DATA_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**9   # silence the "longer than 1024 tokens" warning
    out = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint16, shape=(n_tokens,))
    filled, start = 0, time.time()
    lines = lzma.open(urllib.request.urlopen(CC100_URL), "rt", encoding="utf-8")
    batch = []
    for line in lines:
        batch.append(line.strip())
        if len(batch) < 20_000:
            continue
        ids = tokenizer(batch)["input_ids"]
        for text, piece in zip(batch, ids):
            piece = piece if text else [EOS]
            n = min(len(piece), n_tokens - filled)
            out[filled : filled + n] = piece[:n]
            filled += n
        batch = []
        print(f"\rtokenized {filled / 1e6:.1f}M / {n_tokens / 1e6:.0f}M tokens "
              f"({time.time() - start:.0f}s)", end="", flush=True)
        if filled >= n_tokens:
            break
    print()
    assert filled == n_tokens, f"stream ended after {filled} tokens"
    out.flush()
    del out
    path.with_suffix(".tmp.npy").rename(path)
    return path


def load_rows(path: Path, ctx: int) -> tuple[t.Tensor, t.Tensor]:
    """Train and validation rows of ctx tokens, as int32 tensors (convert with .long() before use)."""
    tokens = np.load(path, mmap_mode="r")
    train, val = tokens[:-VAL_TOKENS], tokens[-VAL_TOKENS:]
    to_rows = lambda a: t.from_numpy(a[: len(a) // ctx * ctx].astype(np.int32)).view(-1, ctx)
    return to_rows(train), to_rows(val)


# ---------------------------------------------------------------- models

class ZeroMLP(nn.Module):
    """Stands in for a removed MLP: the block then adds nothing after attention."""
    def forward(self, x):
        return t.zeros_like(x)


def build_model(args) -> nn.Module:
    """The model described by args, with block i of the unrolled stack tied to block i % n_layers."""
    t.manual_seed(args.seed)
    n_blocks = args.n_layers * args.n_loops
    if args.arch == "mine":
        cfg = Config(d_model=args.d_model, d_vocab=VOCAB, n_ctx=args.ctx, d_head=args.d_model // args.n_heads,
                     n_heads=args.n_heads, n_layers=args.n_layers, n_loops=args.n_loops, tokenizer_name="gpt2")
        return LoopedAttnOnly(cfg)

    # n_layer is the unrolled depth, so GPT-2's init (which scales some weights by depth) treats
    # the looped twin and the control the same way
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=args.ctx, n_embd=args.d_model, n_layer=n_blocks,
                     n_head=args.n_heads, resid_pdrop=args.dropout, embd_pdrop=args.dropout,
                     attn_pdrop=args.dropout, bos_token_id=EOS, eos_token_id=EOS, use_cache=False)
    model = GPT2LMHeadModel(cfg)
    blocks = model.transformer.h
    for b in blocks:
        if args.no_mlp:
            b.mlp, b.ln_2 = ZeroMLP(), nn.Identity()
        if args.no_ln:
            b.ln_1, b.ln_2 = nn.Identity(), nn.Identity()
    if args.no_ln:
        model.transformer.ln_f = nn.Identity()
    for i in range(args.n_layers, n_blocks):
        blocks[i] = blocks[i % args.n_layers]   # the same module object: shared weights
    return model


def get_logits(model, tokens):
    return model(tokens) if isinstance(model, LoopedAttnOnly) else model(tokens).logits


def attention_patterns(model, tokens) -> list[t.Tensor]:
    """One [batch, head, dest, src] pattern per unrolled block, in order (loop-major)."""
    if isinstance(model, LoopedAttnOnly):
        store = []
        for b in model.blocks:
            b.attn.pattern_store = store
        try:
            model(tokens)
        finally:
            for b in model.blocks:
                b.attn.pattern_store = None
        return store
    model.set_attn_implementation("eager")   # the faster default (sdpa) doesn't return patterns
    try:
        return list(model(tokens, output_attentions=True).attentions)
    finally:
        model.set_attn_implementation("sdpa")


def check_tied(model, args):
    """The unrolled blocks are the same modules, so the loops can't drift apart."""
    if isinstance(model, LoopedAttnOnly):
        return
    blocks = model.transformer.h
    for i in range(args.n_layers, len(blocks)):
        assert blocks[i] is blocks[i % args.n_layers], f"block {i} is no longer tied"


def check_tied_state(state_dict: dict, args):
    """In a saved state dict, every h.i.* tensor must equal its h.(i % n_layers).* counterpart."""
    for k, v in state_dict.items():
        parts = k.split(".")
        if parts[:2] == ["transformer", "h"] and int(parts[2]) >= args.n_layers:
            src = ".".join(parts[:2] + [str(int(parts[2]) % args.n_layers)] + parts[3:])
            assert t.equal(v, state_dict[src]), f"{k} differs from {src}"


def load_run(run_dir: str | Path, step: int | None = None):
    """Rebuild a run's model (blocks tied again) and load its weights. step=None means the latest."""
    run_dir = Path(run_dir)
    args = argparse.Namespace(**json.loads((run_dir / "train_args.json").read_text()))
    path = sorted(run_dir.glob("step_*.pt"))[-1] if step is None else run_dir / f"step_{step:06d}.pt"
    ckpt = t.load(path, map_location="cpu")
    model = build_model(args)
    model.load_state_dict(ckpt["state_dict"])   # both names of a shared tensor load the same values
    check_tied(model, args)
    model.eval()
    return args, model


# ---------------------------------------------------------------- loss and evaluation

def lm_loss(logits, tokens, reduction="mean"):
    """Next-token cross-entropy, leaving out end-of-text targets as Aoyama et al.'s collator does."""
    targets = tokens[:, 1:].clone()
    targets[targets == EOS] = -100
    return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), targets.flatten(),
                           ignore_index=-100, reduction=reduction)


def make_ps_tokens(n=100, half=50, seed=0) -> t.Tensor:
    """Aoyama et al.'s test input: `half` distinct random tokens, repeated twice, no BOS."""
    g = t.Generator().manual_seed(seed)
    x = t.stack([t.randperm(VOCAB, generator=g)[:half] for _ in range(n)])
    return t.cat([x, x], dim=1)


@t.no_grad()
def evaluate(model, val_rows, ps_tokens, args, device) -> dict:
    model.eval()
    half = ps_tokens.shape[1] // 2

    val = [lm_loss(get_logits(model, val_rows[i : i + args.batch_size].to(device).long()),
                   val_rows[i : i + args.batch_size].to(device).long()).item()
           for i in range(0, args.val_rows, args.batch_size)]

    # dest p in the second copy; induction source = p - (half - 1), previous token = p - 1
    rep = ps_tokens.to(device)
    dest = t.arange(half, 2 * half, device=device)
    ind, prev = [], []
    for p in attention_patterns(model, rep):
        ind.append(p[:, :, dest, dest - (half - 1)].mean(dim=(0, 2)).tolist())
        prev.append(p[:, :, dest, dest - 1].mean(dim=(0, 2)).tolist())
    # loss index i predicts token i+1; index half-1 predicts the second copy's first token, which
    # can't be known, so the second-copy (predictable) losses start at index half
    rep_loss = lm_loss(get_logits(model, rep), rep, reduction="none").view(len(rep), -1)

    flat = [(s, f"{blk}.{h}") for blk, row in enumerate(ind) for h, s in enumerate(row)]
    best, best_head = max(flat)
    model.train()
    return {"val_loss": sum(val) / len(val),
            "rep_first_half": rep_loss[:, : half - 1].mean().item(),
            "rep_second_half": rep_loss[:, half:].mean().item(),
            "ind_max": best, "ind_head": best_head,
            "prev_max": max(max(row) for row in prev),
            "ind": [[round(s, 4) for s in row] for row in ind],
            "prev": [[round(s, 4) for s in row] for row in prev]}


# ---------------------------------------------------------------- phase checkpoints

class PhaseCheckpointer:
    """Decides, from the induction score at each evaluation, which checkpoints to save.

    pre-formation: the induction score's crossing of pre_threshold can't be foreseen, so a single
        "latest" checkpoint is kept, overwritten every rolling_steps; at the crossing it is kept
        for good (renamed), and no more "latest" saves happen.
    onset, mid-transition: saved at the first evaluation whose score exceeds their threshold.
    (end is saved by the training loop.)

    update() only returns actions, (kind, label, step), so it can be tested without a model:
        ("save_latest", None, step)        overwrite the latest checkpoint
        ("keep_latest", label, its_step)   keep the latest checkpoint under its step's name
        ("save", label, step)              save a checkpoint now
    """

    def __init__(self, pre_threshold: float, thresholds: dict[str, float], rolling_steps: int):
        self.pre_threshold, self.thresholds, self.rolling_steps = pre_threshold, thresholds, rolling_steps
        self.latest_step: int | None = None
        self.pre_done = False
        self.done: set[str] = set()

    def update(self, step: int, ind: float) -> list[tuple[str, str | None, int]]:
        actions = []
        if not self.pre_done:
            if ind > self.pre_threshold:
                self.pre_done = True
                if self.latest_step is not None:
                    actions.append(("keep_latest", "pre-formation", self.latest_step))
            elif self.latest_step is None or step - self.latest_step >= self.rolling_steps:
                self.latest_step = step
                actions.append(("save_latest", None, step))
        for label, threshold in self.thresholds.items():
            if label not in self.done and ind > threshold:
                self.done.add(label)
                actions.append(("save", label, step))
        return actions


# ---------------------------------------------------------------- main

def get_device(name):
    if name is not None:
        return t.device(name)
    if t.cuda.is_available():
        return t.device("cuda")
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["gpt2", "mine"], default="gpt2")
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_loops", type=int, default=1)
    p.add_argument("--no_mlp", action="store_true")
    p.add_argument("--no_ln", action="store_true")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--schedule_tokens", type=float, default=1e9, help="cosine schedule length (their 1B)")
    p.add_argument("--max_tokens", type=float, default=100e6, help="stop training after this many")
    p.add_argument("--data_tokens", type=int, default=110_000_000, help="how much CC100 to prepare")
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--val_rows", type=int, default=32)
    p.add_argument("--ckpt_mtokens", default="5,10,20,30,40,50,60,80,100",
                   help="save weights after these many million tokens (and at the end)")
    p.add_argument("--ckpt_phases", default=None,
                   help="instead of --ckpt_mtokens: three induction-score thresholds, e.g. 0.03,0.1,0.25, "
                        "giving pre-formation, onset and mid-transition checkpoints (plus the end)")
    p.add_argument("--latest_mtokens", type=float, default=5,
                   help="with --ckpt_phases: how often the pre-formation candidate is overwritten")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA only)")
    p.add_argument("--device", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--prepare_only", action="store_true")
    args = p.parse_args()

    data_path = prepare_cc100(args.data_tokens)
    if args.prepare_only:
        print(f"data ready: {data_path}")
        return
    if args.arch == "mine" and (args.no_mlp or args.no_ln):
        print("note: --arch mine has no MLP, LayerNorm or dropout; --no_mlp, --no_ln and --dropout are ignored")

    tag = args.arch + ("-nomlp" if args.no_mlp else "") + ("-noln" if args.no_ln else "") \
        + (f"-drop{args.dropout:g}" if args.arch == "gpt2" and args.dropout != 0.1 else "")
    run_name = args.run_name or f"rep_{tag}_L{args.n_layers}x{args.n_loops}_b{args.batch_size}_c{args.ctx}_s{args.seed}"
    run_dir = CKPT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    device = get_device(args.device)

    train_rows, val_rows = load_rows(data_path, args.ctx)
    tokens_per_step = args.batch_size * args.ctx
    max_steps = int(args.max_tokens // tokens_per_step)
    schedule_steps = int(args.schedule_tokens // tokens_per_step)
    predicted = math.exp(13.26) * args.batch_size ** -0.37 * args.ctx ** -0.62
    print(f"{len(train_rows)} training rows ({len(train_rows) * args.ctx / 1e6:.0f}M tokens); "
          f"{max_steps} steps planned, cosine schedule over {schedule_steps}; "
          f"Aoyama law predicts induction heads around step {predicted:.0f}")
    if max_steps * args.batch_size > len(train_rows):
        print("note: run is longer than the data, so some rows will be seen more than once")

    model = build_model(args).to(device)   # tied before the optimizer is made
    check_tied(model, args)
    n_params = sum(q.numel() for q in model.parameters())   # shared tensors count once
    print(f"device={device}  {n_params:,} parameters")
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = t.optim.AdamW(
        [{"params": [q for q in params if q.ndim >= 2], "weight_decay": args.weight_decay},
         {"params": [q for q in params if q.ndim < 2], "weight_decay": 0.0}],   # no decay on biases/LN, as HF
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(schedule_steps * 0.01), schedule_steps)

    ps_tokens = make_ps_tokens()
    if args.ckpt_phases:
        pre, onset, mid = (float(x) for x in args.ckpt_phases.split(","))
        phaser = PhaseCheckpointer(pre, {"onset": onset, "mid-transition": mid},
                                   int(args.latest_mtokens * 1e6 // tokens_per_step))
        ckpt_steps = {max_steps}
    else:
        phaser = None
        ckpt_steps = {int(float(m) * 1e6 // tokens_per_step) for m in args.ckpt_mtokens.split(",")} | {max_steps}
    phases = {}   # label -> step, written to phases.json

    def save(path):
        sd = model.state_dict()
        check_tied_state(sd, args)
        t.save({"step": step, "state_dict": sd}, path)

    def record(label, s):
        phases[label] = s
        (run_dir / "phases.json").write_text(json.dumps(phases, indent=2))

    gen = t.Generator().manual_seed(args.seed)
    order, pos = t.randperm(len(train_rows), generator=gen), 0
    metrics_file = open(run_dir / "metrics.jsonl", "w")
    ema, start = None, time.time()
    model.train()
    for step in range(max_steps + 1):
        if step % args.eval_every == 0 or step == max_steps:
            m = evaluate(model, val_rows, ps_tokens, args, device)
            m.update(step=step, tokens=step * tokens_per_step, lr=scheduler.get_last_lr()[0],
                     train_loss_ema=ema, minutes=(time.time() - start) / 60)
            metrics_file.write(json.dumps(m) + "\n")
            metrics_file.flush()
            print(f"step {step:6d} | {step * tokens_per_step / 1e6:6.1f}M tok | train {ema or float('nan'):.3f} | "
                  f"val {m['val_loss']:.3f} | induction {m['ind_max']:.3f} ({m['ind_head']}) | "
                  f"prev-token {m['prev_max']:.3f} | rep loss {m['rep_first_half']:.2f} -> "
                  f"{m['rep_second_half']:.2f} | {m['minutes']:.1f} min", flush=True)
            for kind, label, s in (phaser.update(step, m["ind_max"]) if phaser else []):
                if kind == "save_latest":
                    save(run_dir / "latest.pt")
                elif kind == "keep_latest":
                    (run_dir / "latest.pt").rename(run_dir / f"step_{s:06d}.pt")
                    record(label, s)
                else:
                    save(run_dir / f"step_{s:06d}.pt")
                    record(label, s)
                print(f"  checkpoint: {kind} {label or ''} at step {s}", flush=True)
        if step in ckpt_steps:
            save(run_dir / f"step_{step:06d}.pt")
            if phaser:
                record("end", step)
                (run_dir / "latest.pt").unlink(missing_ok=True)   # never needed: no crossing happened
        if step == max_steps:
            break

        if pos + args.batch_size > len(order):
            order, pos = t.randperm(len(train_rows), generator=gen), 0
        batch = train_rows[order[pos : pos + args.batch_size]].to(device).long()
        pos += args.batch_size
        with t.autocast("cuda", dtype=t.bfloat16, enabled=args.bf16 and device.type == "cuda"):
            loss = lm_loss(get_logits(model, batch), batch)
        optimizer.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        check_tied(model, args)
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()

    metrics_file.close()
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
