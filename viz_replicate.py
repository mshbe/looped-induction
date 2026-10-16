# %% [markdown]
# # Attention patterns of replicate.py checkpoints, on repeated random tokens
#
# Run cell by cell in VS Code (shift-return) with the project's .venv as the kernel, or as a script:
#     uv run python viz_replicate.py checkpoints/rep_gpt2_L1x2_300M_s3 [more run dirs...]
#
# For each run, every checkpoint in phases.json (pre-formation, onset, mid-transition, any milestones, end) is loaded, in step order;
# runs without phases.json use all their checkpoints. Outputs go to figures/<run>/:
#     overview.html            every head at every checkpoint, one small heatmap each (hover for tokens)
#     <label>_<block>.html     circuitsvis grid for one checkpoint and one block (or loop)
# and a per-head score table is printed.
#
# The input is the induction test's format: random distinct tokens, then the same tokens again, no BOS.
# An induction head shows a stripe in the second half, (half - 1) positions below the main diagonal;
# a previous-token head shows the stripe just below the diagonal.
# Block labels: "loop{k}" for a looped run (all loops are the same weights), "layer{l}" otherwise.

# %%
import json
import sys
from pathlib import Path

import circuitsvis as cv
import plotly.graph_objects as go
import torch as t
from plotly.subplots import make_subplots
from transformers import AutoTokenizer

from replicate import attention_patterns, load_run, make_ps_tokens

FIG_DIR = Path("figures")
DISPLAY_HALF = 20     # display sequence: 20 random tokens, repeated (40 positions)
SCORE_HALF = 50       # scores use the same test as training: 100 sequences of 50 tokens, repeated
# sequential blue ramp, near-white at zero so empty attention recedes
COLORSCALE = [[0.0, "#ffffff"], [0.05, "#cde2fb"], [0.2, "#86b6ef"], [0.4, "#3987e5"],
              [0.6, "#256abf"], [0.8, "#184f95"], [1.0, "#0d366b"]]

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def checkpoints(run_dir: Path) -> list[tuple[str, int]]:
    """(label, step) pairs: the phases.json checkpoints in phase order, else every saved step."""
    phases_file = run_dir / "phases.json"
    if phases_file.exists():
        phases = json.loads(phases_file.read_text())   # may also hold milestones like "300M"
        return sorted(phases.items(), key=lambda kv: kv[1])
    steps = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("step_*.pt"))
    return [(f"step{s}", s) for s in steps]


def block_labels(args) -> list[str]:
    n = args.n_layers * args.n_loops
    if args.n_loops == 1:
        return [f"layer{i}" for i in range(n)]
    if args.n_layers == 1:
        return [f"loop{i}" for i in range(n)]
    return [f"loop{i // args.n_layers}.layer{i % args.n_layers}" for i in range(n)]


@t.no_grad()
def head_scores(model) -> tuple[t.Tensor, t.Tensor]:
    """Induction and previous-token scores, [block, head], computed as in replicate.evaluate."""
    rep = make_ps_tokens(n=100, half=SCORE_HALF)
    dest = t.arange(SCORE_HALF, 2 * SCORE_HALF)
    pats = attention_patterns(model, rep)
    ind = t.stack([p[:, :, dest, dest - (SCORE_HALF - 1)].mean(dim=(0, 2)) for p in pats])
    prev = t.stack([p[:, :, dest, dest - 1].mean(dim=(0, 2)) for p in pats])
    return ind, prev


def token_strings(ids: t.Tensor) -> list[str]:
    return [tokenizer.decode([i]).replace("\n", "\\n") for i in ids.tolist()]


# %%
def analyse_run(run_dir: Path):
    run_dir = Path(run_dir)
    out_dir = FIG_DIR / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    display = make_ps_tokens(n=1, half=DISPLAY_HALF, seed=1)
    strs = token_strings(display[0])

    results = []   # (label, step, labels, patterns [block][head, dest, src], ind, prev)
    for label, step in checkpoints(run_dir):
        args, model = load_run(run_dir, step)
        with t.no_grad():
            pats = [p[0] for p in attention_patterns(model, display)]
        ind, prev = head_scores(model)
        blocks = block_labels(args)
        results.append((label, step, blocks, pats, ind, prev))

        for b, (name, pat) in enumerate(zip(blocks, pats)):
            html = cv.attention.attention_heads(
                attention=pat, tokens=strs,
                attention_head_names=[f"{name}.H{h} ind {ind[b, h]:.2f} prev {prev[b, h]:.2f}"
                                      for h in range(pat.shape[0])])
            (out_dir / f"{label}_{name}.html").write_text(str(html))

        tokens_m = step * args.batch_size * args.ctx / 1e6
        print(f"\n{run_dir.name}  {label}  step {step} ({tokens_m:.0f}M tokens)")
        print("        " + "".join(f"{'H' + str(h):>12}" for h in range(ind.shape[1])))
        for b, name in enumerate(blocks):
            print(f"{name:>8}" + "".join(f"  {ind[b, h]:.2f}/{prev[b, h]:.2f}" for h in range(ind.shape[1]))
                  + "   (induction/prev-token)")

    overview(run_dir.name, results, strs, out_dir / "overview.html")
    print(f"\nsaved figures to {out_dir}/")
    return results


def overview(run_name, results, strs, path):
    """One small heatmap per (checkpoint, block) row and head column, on a shared 0-1 scale."""
    n_heads = results[0][3][0].shape[0]
    rows = [(label, step, b, name) for label, step, blocks, *_ in results for b, name in enumerate(blocks)]
    titles = []
    for label, step, blocks, pats, ind, prev in results:
        for b in range(len(blocks)):
            titles += [f"H{h}  ind {ind[b, h]:.2f} · prev {prev[b, h]:.2f}" for h in range(n_heads)]
    fig = make_subplots(rows=len(rows), cols=n_heads, subplot_titles=titles,
                        horizontal_spacing=0.012, vertical_spacing=0.035 if len(rows) <= 8 else 0.02)
    n = len(strs)
    hover = [[[strs[q], strs[k]] for k in range(n)] for q in range(n)]
    r = 0
    for label, step, blocks, pats, ind, prev in results:
        for b, name in enumerate(blocks):
            r += 1
            for h in range(n_heads):
                z = pats[b][h].float().numpy().copy()
                z[t.triu(t.ones(n, n, dtype=t.bool), 1).numpy()] = float("nan")   # future: blank
                fig.add_trace(go.Heatmap(
                    z=z, zmin=0, zmax=1, colorscale=COLORSCALE, showscale=(r == 1 and h == 0),
                    colorbar=dict(title="attention", thickness=12, len=0.3, y=1, yanchor="top"),
                    customdata=hover, xgap=0, ygap=0,
                    hovertemplate=(f"{label} {name} H{h}<br>from %{{y}} '%{{customdata[0]}}'"
                                   "<br>to %{x} '%{customdata[1]}'<br>attention %{z:.2f}<extra></extra>")),
                    row=r, col=h + 1)
            fig.update_yaxes(title_text=f"{label}<br>{name}", title_font_size=11, row=r, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, constrain="domain")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed",
                     scaleanchor="x", constrain="domain")
    for ann in fig.layout.annotations:
        ann.font.size = 9
    fig.update_layout(
        title=dict(text=f"{run_name}: attention on repeated random tokens "
                        f"({DISPLAY_HALF} tokens, then the same {DISPLAY_HALF}); "
                        "scores from the 100-sequence test", font_size=14),
        width=170 * n_heads + 120, height=175 * len(rows) + 90,
        template="plotly_white", margin=dict(l=90, r=60, t=80, b=20))
    fig.write_html(path, include_plotlyjs="cdn")


# %%
if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(str(p.parent) for p in Path("checkpoints").glob("rep_*/phases.json"))
    for run in runs:
        analyse_run(Path(run))
