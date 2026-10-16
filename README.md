**Induction heads in looped 1-layer transformers**

## Summary
This project investigates whether induction heads can form in weight-tied transformers with 2 layers with identical weights ("1 layer 2 loop" transformers). We use a GPT2-style model architecture. Induction heads are part of a relatively simple, very common 2-layer circuit which aids in bigram completion; that is, if AB appear early in input, and A appears later, induction heads boost B in the logits for A. Induction heads do not form in 1 layer transformers, but do form in 2 layer non-weight-tied transformers. We verify that induction heads do form in 1-layer 2-loop models, so what matters is passing through 2 layers total rather than having different weights in the layers. We also trace the induction circuit in both weight-tied and non-weight-tied models, and find the mechanism to be very similar.

## Results
We trained a GPT2-style model with tied weights (45.5M parameters, 7M in transformer blocks) and with untied weights (54M parameters, 14M in transformer blocks) as control. In training, both the tied model and the control form induction heads:

<img width="1700" height="952" alt="image" src="https://github.com/user-attachments/assets/ca327c30-c94f-4089-baa3-aa94912ea2e7" />

This plot shows mean attention of a particular head towards the "correct" token over the course of the training run. That is, the token the head should attend to if it is an induction head.

We see the characteristic induction-circuit attention patterns in the tied mode with input (random tokens, same random tokens: a previous-token attention head in Layer 0 (attention on the subdiagonal) and an induction head in Layer 1 (once repeat starts, attention on the shifted subdiagonal).

<img width="2380" height="1224" alt="image" src="https://github.com/user-attachments/assets/2a463c14-fa4f-4afe-83d4-e5a2edfe49b8" />

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python replicate.py --prepare_only --data_tokens 310000000   # stream + tokenize CC100 into data/
```

## Usage

```bash
# 2-layer control; for the looped model use --n_layers 1 --n_loops 2;
# for the size-matched control add --n_heads 4 --d_head 96 --d_mlp 1536
uv run python replicate.py --n_layers 2 --n_loops 1 --seed 1 --bf16 \
    --data_tokens 310000000 --max_tokens 300e6 --ckpt_phases 0.03,0.1,0.25
```

Each run writes `checkpoints/<run_name>/` (config, metrics every 50 steps, checkpoints).
To analyse a run, set `RUN` in `analysis.ipynb` and run all cells; `viz_replicate.py <run_dir>`
plots its attention patterns.

## Repository

| File | Contents |
|---|---|
| `replicate.py` | Data, models (GPT-2, optionally looped), training, evaluation |
| `circuits.py` | Loads a run as a TransformerLens model |
| `analysis.ipynb` | Per-run analysis notebook |
| `viz_replicate.py` | Attention-pattern plots |
| `model.py`, `training.py`, `convert.py`, `attention_viz.py` | Earlier attention-only experiments |

## References

- Aoyama, Wilcox, Schneider. *Predicting the Emergence of Induction Heads in Language Model Pretraining.*
  [arXiv:2511.16893](https://arxiv.org/abs/2511.16893). Training setup and induction score follow this paper.
- [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0), Chapter 1.2. Basis for the analysis notebook.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
- CC100 data: [data.statmt.org/cc-100](https://data.statmt.org/cc-100/).



