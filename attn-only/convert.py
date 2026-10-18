"""Load a trained LoopedAttnOnly checkpoint and copy its weights into a HookedTransformer.

The HookedTransformer gets one block per (loop, layer) pair, i.e. n_loops * n_layers blocks,
and every copy of a layer gets the same weights. Then cache["pattern", i] is block i's attention.

Usage:
    from convert import load_run, to_hooked
    cfg, model = load_run("checkpoints/L1_loops2_seed0")
    tl_model = to_hooked(model)

Check the conversion:
    uv run python convert.py checkpoints/L1_loops2_seed0
"""

import json
import sys
from pathlib import Path

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from model import Config, LoopedAttnOnly
from training import load_val_tokens


# ---------------------------------------------------------------- loading 

def load_run(run_dir: str | Path, step: int | None = None) -> tuple[Config, LoopedAttnOnly]:
    """Load a training run's Config and weights. step=None means the latest checkpoint."""
    run_dir = Path(run_dir)
    cfg = Config(**json.loads((run_dir / "config.json").read_text()))
    if step is None:
        ckpt_path = sorted(run_dir.glob("step_*.pt"))[-1]
    else:
        ckpt_path = run_dir / f"step_{step:06d}.pt"
    ckpt = t.load(ckpt_path, map_location="cpu")
    model = LoopedAttnOnly(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded {ckpt_path} (step {ckpt['step']})")
    return cfg, model


# ---------------------------------------------------------------- conversion (fill in)

def make_tl_config(cfg: Config, n_loops: int) -> HookedTransformerConfig:
    """A HookedTransformerConfig that computes the same function as LoopedAttnOnly."""
    return HookedTransformerConfig(
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        n_ctx=cfg.n_ctx,
        d_vocab=cfg.d_vocab,
        n_layers=cfg.n_layers*n_loops, #n_loops untied from cfg so can repeat loops dif number of times
        attention_dir="causal",
        attn_only=True,
        normalization_type=None,
        positional_embedding_type="shortformer",
        tokenizer_name=cfg.tokenizer_name,
        device="cpu",
        use_attn_result=True,  # caches each head's output separately (handy for analysis)
    )


def to_hooked(model: LoopedAttnOnly, n_loops: int | None = None) -> HookedTransformer:
    """Unroll the loops into a HookedTransformer. n_loops defaults to the number used in training."""
    cfg = model.cfg
    n_loops = cfg.n_loops if n_loops is None else n_loops
    tl_model = HookedTransformer(make_tl_config(cfg, n_loops))

    # Start from TransformerLens's own state dict, so every key it expects (including its
    # biases and buffers like the causal mask) is present; then overwrite the weights.
    sd = tl_model.state_dict()

    # TODO: embeddings and unembedding. Key names in sd: "embed.W_E", "pos_embed.W_pos", "unembed.W_U".
    #   Print sorted(sd.keys()) once to see all the names.
    sd["embed.W_E"] = model.embed.W_E
    sd["pos_embed.W_pos"]=model.pos_embed.W_pos
    sd["unembed.W_U"]=model.unembed.W_U

    # TODO: attention weights. Your model's block `layer` runs on every loop.
    #   Which TransformerLens block index does (loop, layer) correspond to?
    for loop in range(n_loops):
        for layer in range(cfg.n_layers):
            i = cfg.n_layers*loop + layer
            for name in ["W_Q", "W_K", "W_V", "W_O"]:
                sd[f"blocks.{i}.attn.{name}"] = getattr(model.blocks[layer].attn, name)

    # TODO: biases. Your model has none, so every bias in sd should be zero.
    #   (They're probably already zero at initialization, but don't rely on it.)
    #   Bias keys: blocks.{i}.attn.b_Q / b_K / b_V / b_O, and unembed.b_U.
    for key in sd:
        if ".b_" in key:          # every bias: blocks.{i}.attn.b_Q/K/V/O and unembed.b_U
            sd[key] = t.zeros_like(sd[key])

    tl_model.load_state_dict(sd)
    tl_model.eval()
    return tl_model

# ---------------------------------------------------------------- the check (fill in)

@t.no_grad()
def check_conversion(model: LoopedAttnOnly, tl_model: HookedTransformer, tokens: t.Tensor) -> None:
    """Both models should give the same logits on the same tokens, up to float error."""
    # TODO: run both models on `tokens` and compare the logits.
    #   Print the largest absolute difference, then assert it is small
    #   (t.testing.assert_close(a, b, atol=1e-4, rtol=1e-4) does this and reports where they differ).
    #   Note: give tl_model the token tensor, not a string, so it doesn't add a BOS token.
    loop_out=model(tokens)
    tl_out=tl_model(tokens)

    #
    # TODO (optional): also check the cache has one attention pattern per unrolled block:
    #   _, cache = tl_model.run_with_cache(tokens)
    #   cache["pattern", i] should have shape [batch, n_heads, seq, seq] for each block i.
    print(f"max |logit difference|: {(loop_out - tl_out).abs().max().item():.2e}")
    t.testing.assert_close(loop_out, tl_out, atol=1e-4, rtol=1e-4)
    return 


if __name__ == "__main__":
    run_dir = sys.argv[1]
    cfg, model = load_run(run_dir)
    tl_model = to_hooked(model)

    # a few rows of real text from the TinyStories validation data, plus some random tokens
    text = load_val_tokens(tl_model.tokenizer, cfg.n_ctx)[:4].long()
    rand = t.randint(0, cfg.d_vocab, (4, cfg.n_ctx))
    check_conversion(model, tl_model, text)
    check_conversion(model, tl_model, rand)
    print("conversion check passed")
