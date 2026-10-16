"""Load a replicate.py run as a TransformerLens model, for ARENA-style circuit analysis.

The Hugging Face GPT-2 is unrolled into n_layers * n_loops TransformerLens blocks; a looped run's
blocks get copies of the same weights, so block i is (loop i // n_layers, layer i % n_layers).
LayerNorm is folded into the neighbouring weights (as TransformerLens does for pretrained models):
each LayerNorm becomes a pure "subtract mean, divide by norm" step, so the circuits are linear apart
from that per-token scale. Writing weights are centred, so the mean subtraction does nothing.

Usage:
    from circuits import load_analysis
    a = load_analysis("checkpoints/rep_gpt2_L1x2_300M_s3")   # final checkpoint
    a.tl, a.args, a.metrics, a.phases, a.labels
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import convert_gpt2_weights

from replicate import VOCAB, load_run, make_ps_tokens


@dataclass
class Analysis:
    run_dir: Path
    step: int
    args: object              # the run's training arguments
    hf: object                # the Hugging Face model as trained
    tl: HookedTransformer     # unrolled, LayerNorm folded
    metrics: list[dict]       # metrics.jsonl, one dict per evaluation
    phases: dict              # phases.json (label -> step), or {} for runs without it
    labels: list[str]         # a name per unrolled block: "loop0", "layer1", ...


def block_labels(args) -> list[str]:
    n = args.n_layers * args.n_loops
    if args.n_loops == 1:
        return [f"layer{i}" for i in range(n)]
    if args.n_layers == 1:
        return [f"loop{i}" for i in range(n)]
    return [f"loop{i // args.n_layers}.layer{i % args.n_layers}" for i in range(n)]


def to_hooked(hf, args) -> HookedTransformer:
    """Unrolled TransformerLens copy of a replicate.py GPT-2 (MLP and LayerNorm required)."""
    assert args.arch == "gpt2" and not args.no_mlp and not args.no_ln, "only the full GPT-2 runs are supported"
    cfg = HookedTransformerConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers * args.n_loops,
        d_head=getattr(args, "d_head", None) or args.d_model // args.n_heads,
        d_mlp=getattr(args, "d_mlp", None) or 4 * args.d_model, act_fn="gelu_new",
        n_ctx=args.ctx, d_vocab=VOCAB, normalization_type="LNPre",   # LNPre: weights folded in
        positional_embedding_type="standard", default_prepend_bos=False,
        tokenizer_name="gpt2", original_architecture="GPT2LMHeadModel", device="cpu")
    tl = HookedTransformer(cfg)
    tl.load_and_process_state_dict(convert_gpt2_weights(hf, cfg), fold_ln=True, center_writing_weights=True,
                                   center_unembed=True, fold_value_biases=True)
    tl.eval()
    return tl


@t.no_grad()
def check_same(hf, tl, atol=1e-3):
    """Both models must give the same next-token log-probabilities (centring the unembedding
    shifts every logit at a position by the same amount, so logits themselves differ)."""
    x = make_ps_tokens(n=4, half=50)
    diff = (hf(x).logits.log_softmax(-1) - tl(x).log_softmax(-1)).abs().max().item()
    assert diff < atol, f"TransformerLens copy differs from the trained model by {diff:.1e}"
    return diff


def load_analysis(run_dir, step: int | None = None) -> Analysis:
    run_dir = Path(run_dir)
    if step is None:
        step = max(int(p.stem.split("_")[1]) for p in run_dir.glob("step_*.pt"))
    args, hf = load_run(run_dir, step)
    tl = to_hooked(hf, args)
    check_same(hf, tl)
    metrics = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    phases_file = run_dir / "phases.json"
    phases = json.loads(phases_file.read_text()) if phases_file.exists() else {}
    return Analysis(run_dir, step, args, hf, tl, metrics, phases, block_labels(args))
