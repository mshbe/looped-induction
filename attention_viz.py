# %% [markdown]
# # Attention patterns and head scores
#
# Run cell by cell in VS Code (shift-return), with the project's .venv selected as the kernel.
# Each attention grid is also saved as an HTML file in figures/ that opens in any browser.
#
# Block labels: in the unrolled TransformerLens model, block i is (loop, layer) with
# i = loop * n_layers + layer. Labels below read "loop{loop}.L{layer}.H{head}".

# %%
import json
from pathlib import Path

import circuitsvis as cv
import plotly.graph_objects as go
import torch as t

from convert import load_run, to_hooked
from training import common_token_ids, load_val_tokens, make_repeated_tokens

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

RUN = "checkpoints/ts_L1_loops2_seed0"   # change to the run you want to look at
STEP = None                            # None = latest checkpoint

cfg, model = load_run(RUN, STEP)
tl_model = to_hooked(model)
n_blocks = cfg.n_layers * cfg.n_loops
half_len = (cfg.n_ctx - 1) // 2


def block_label(i: int) -> str:
    return f"loop{i // cfg.n_layers}.L{i % cfg.n_layers}"


# %% [markdown]
# ## Inputs: a stretch of real text, and repeated random tokens

# %%
val_tokens = load_val_tokens(tl_model.tokenizer, cfg.n_ctx).long()
text_tokens = val_tokens[:1]   # one validation row (never seen in training)

# random tokens for the repeated sequences come from the same common-token set used in training
train_args = json.loads((Path(RUN) / "train_args.json").read_text())
rep_ids = common_token_ids(val_tokens, cfg.d_vocab, train_args.get("rep_vocab_size", 2000),
                           tl_model.tokenizer.all_special_ids)
rep_tokens = make_repeated_tokens(tl_model.tokenizer.bos_token_id, rep_ids, half_len, batch=16)

_, text_cache = tl_model.run_with_cache(text_tokens)
_, rep_cache = tl_model.run_with_cache(rep_tokens)


# %% [markdown]
# ## Head scores (ARENA-style detectors)
#
# Each score is the average attention a head puts on one particular source position:
# - previous-token: from each position to the one just before it (measured on real text)
# - duplicate-token: from a token in the second copy to the same token in the first copy
# - induction: from a token in the second copy to the token *after* its first-copy occurrence
#
# Scores near 1 mean the head does almost exclusively that; heads with ~0.3+ are worth a look.

# %%
def head_scores(text_cache, rep_cache):
    L = half_len
    rows = []
    for i in range(n_blocks):
        text_pat = text_cache["pattern", i]   # [batch, head, dest, src]
        rep_pat = rep_cache["pattern", i]
        # diagonal(offset=-k) gives pattern[k + j, j]; dest position p = k + j.
        prev = text_pat.diagonal(-1, dim1=-2, dim2=-1).mean(dim=(0, -1))
        dup = rep_pat.diagonal(-L, dim1=-2, dim2=-1)[..., 1:].mean(dim=(0, -1))        # dest p >= L+1
        ind = rep_pat.diagonal(-(L - 1), dim1=-2, dim2=-1)[..., 2:].mean(dim=(0, -1))  # dest p >= L+1
        for h in range(cfg.n_heads):
            rows.append((f"{block_label(i)}.H{h}", prev[h].item(), dup[h].item(), ind[h].item()))
    return rows


print(f"{'head':<16}{'prev-token':>12}{'duplicate':>12}{'induction':>12}")
for name, prev, dup, ind in head_scores(text_cache, rep_cache):
    flag = "  <--" if max(prev, dup, ind) > 0.3 else ""
    print(f"{name:<16}{prev:>12.3f}{dup:>12.3f}{ind:>12.3f}{flag}")


# %% [markdown]
# ## Attention grids (the ARENA-style view)
#
# Hover over a head to see its pattern; hover over a token to see where it attends.
# Real text: look for a diagonal stripe just below the main diagonal (previous-token heads).
# Grids show only the first 40 tokens; the scores above use full-length sequences.

# %%
def show_block(cache, str_tokens, i, name, n_show=40):
    """Grid for block i, first sequence in the batch, first n_show tokens only (full 256 is unreadable)."""
    html = cv.attention.attention_heads(
        attention=cache["pattern", i][0, :, :n_show, :n_show],   # [head, dest, src]
        tokens=str_tokens[:n_show],
        attention_head_names=[f"{block_label(i)}.H{h}" for h in range(cfg.n_heads)],
    )
    path = FIG_DIR / f"{Path(RUN).name}_{name}_{block_label(i)}.html"
    path.write_text(str(html))
    print(f"saved {path}")
    return html


text_str = tl_model.to_str_tokens(text_tokens[0])

# %%
show_block(text_cache, text_str, 0, "text")

# %%
# For a 2-block model; skip this cell if n_blocks == 1
show_block(text_cache, text_str, 1, "text")

# %%
# A short repeated sequence just for display: BOS + 20 random tokens + the same 20 again.
# An induction head shows a stripe 19 positions below the main diagonal in the second half.
short_rep = make_repeated_tokens(tl_model.tokenizer.bos_token_id, rep_ids, 20, batch=1)
_, short_rep_cache = tl_model.run_with_cache(short_rep)
short_rep_str = tl_model.to_str_tokens(short_rep[0])
show_block(short_rep_cache, short_rep_str, n_blocks - 1, "repeated", n_show=41)


# %% [markdown]
# ## Scores across training
#
# For each saved checkpoint: the highest previous-token score and the highest induction
# score over all heads. A previous-token head appearing with no induction head would mean
# the first half of the circuit formed but not the second.

# %%
steps, best_prev, best_ind = [], [], []
for ckpt in sorted(Path(RUN).glob("step_*.pt")):
    step = int(ckpt.stem.split("_")[1])
    _, model_s = load_run(RUN, step)
    tl_s = to_hooked(model_s)
    _, text_c = tl_s.run_with_cache(text_tokens)
    _, rep_c = tl_s.run_with_cache(rep_tokens)
    scores = head_scores(text_c, rep_c)
    steps.append(step)
    best_prev.append(max(s[1] for s in scores))
    best_ind.append(max(s[3] for s in scores))

fig = go.Figure()
fig.add_trace(go.Scatter(x=steps, y=best_prev, name="previous-token (best head)",
                         mode="lines+markers", line=dict(color="#2a78d6", width=2), marker=dict(size=8)))
fig.add_trace(go.Scatter(x=steps, y=best_ind, name="induction (best head)",
                         mode="lines+markers", line=dict(color="#eb6834", width=2), marker=dict(size=8)))
fig.update_layout(title=f"Head scores over training: {Path(RUN).name}",
                  xaxis_title="training step", yaxis_title="score (mean attention)",
                  yaxis_range=[0, 1], hovermode="x unified", template="plotly_white")
fig.write_html(FIG_DIR / f"{Path(RUN).name}_scores_over_training.html")
fig.show()
