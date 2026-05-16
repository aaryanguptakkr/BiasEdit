# pythia-1b — Bias Tracing Report

Generated: 2026-05-16  
Source: `/deepfreeze/share/xuxin_transfer/bias_tracing/results.zip`

## What this report measures

Causal tracing asks: *which (subject token, layer) positions causally mediate bias?*

For each sentence pair (stereotyped vs. anti-stereotyped), the subject tokens are corrupted
with Gaussian noise. Then, one hidden state at a time is restored to its clean value.
The **indirect effect** at (token i, layer j) = how much the prediction recovers when
only that one state is restored. Reported here as NIE (normalized by the clean–corrupted gap).

Three restore conditions per sentence pair:
- **Full restore** (single state): all components (MLP + Attn) restored at that layer
- **MLP-only**: only MLP output restored; Attn output left corrupted
- **Attn-only**: only Attn output restored; MLP output left corrupted

Scores are aggregated over **subject token positions only** (not the full sentence),
then averaged over all sentence pairs in the domain.

### Interpreting the NIE pattern

In factual recall (ROME paper), NIE peaks sharply at **specific mid-layer MLPs** — the
knowledge is "stored" there and computed on demand. Bias may behave differently:

- **NIE highest at L0 and declining**: bias is primarily lexical — it enters through
  the token embedding and is not further computed or concentrated by transformer layers.
  Words like "father" or "Hispanic" carry the stereotypic signal in their embedding itself.
- **NIE goes negative at later layers**: restoring a subject state at a late layer
  creates an *inconsistent* internal state (one clean token among corrupted context),
  which can hurt prediction below the corrupted baseline.
- **MLP-only NIE flat or negative**: the MLP pathway alone does not localize bias,
  unlike factual knowledge where a specific MLP layer is the key mediator.

⚠ **Religion domain**: very few cases (24–44) and often tiny effect gap (< 0.03).
NIE estimates for religion are unreliable — treat with caution.

---

## Field reference

| Field | Description |
|---|---|
| **N cases** | Sentence pairs processed |
| **High score** | Mean model probability on the correct token (clean run) |
| **Low score** | Mean probability after corrupting subject tokens |
| **Effect gap** | High − Low — how much corruption hurts; < 0.03 = low-signal |
| **Peak All/MLP/Attn** | Layer with highest NIE under each restore condition |
| **NIE L0** | Normalized indirect effect at the embedding layer |
| **NIE L-mid** | NIE at the middle layer |
| **NIE L-last** | NIE at the final layer |

---

## Summary table

| Checkpoint | Domain | N | Gap | Peak All | Peak MLP | Peak Attn | NIE L0 | NIE L-mid | NIE L-last |
|---|---|---|---|---|---|---|---|---|---|
| step0 | gender ⚠ | 608 | 0.0094 | 0 | 2 | 2 | -0.62 | -0.91 | -0.95 |
| step0 | profession ⚠ | 496 | 0.0046 | 0 | 2 | 2 | -0.44 | -0.75 | -0.91 |
| step0 | race ⚠ | 886 | 0.0106 | 0 | 4 | 4 | -0.19 | -0.38 | -0.56 |
| step0 | religion ⚠ | 24 | 0.0186 | 0 | 0 | 5 | +0.53 | +0.10 | +0.08 |
| step1k | gender ⚠ | 608 | 0.0164 | 0 | 2 | 2 | -1.12 | -1.56 | -1.59 |
| step1k | profession ⚠ | 496 | 0.0132 | 0 | 3 | 2 | -0.11 | -0.71 | -0.67 |
| step1k | race ⚠ | 886 | 0.0134 | 0 | 5 | 0 | -0.12 | -0.31 | -0.42 |
| step1k | religion ⚠ | 24 | -0.0090 | 11 | 8 | 14 | +0.00 | +0.00 | +0.00 |
| step5k | gender ⚠ | 608 | 0.0299 | 1 | 4 | 5 | -0.37 | -0.64 | -0.84 |
| step5k | profession | 496 | 0.0304 | 0 | 5 | 5 | +0.40 | +0.24 | -0.18 |
| step5k | race | 886 | 0.0318 | 0 | 3 | 5 | +0.07 | -0.21 | -0.27 |
| step5k | religion | 24 | 0.0432 | 0 | 4 | 2 | -1.75 | -1.89 | -1.97 |
| step81k | gender | 608 | 0.0631 | 1 | 1 | 5 | +0.09 | -0.24 | -0.41 |
| step81k | profession | 496 | 0.0471 | 1 | 1 | 5 | +0.42 | +0.10 | -0.15 |
| step81k | race | 886 | 0.1034 | 0 | 2 | 3 | +0.29 | -0.02 | -0.07 |
| step81k | religion | 24 | 0.0789 | 4 | 2 | 1 | -0.87 | -0.86 | -1.08 |
| step137k | gender | 608 | 0.0748 | 0 | 4 | 5 | +0.12 | -0.22 | -0.36 |
| step137k | profession | 496 | 0.0587 | 0 | 0 | 3 | +0.50 | +0.24 | -0.04 |
| step137k | race | 886 | 0.1127 | 1 | 1 | 3 | +0.28 | -0.00 | -0.05 |
| step137k | religion | 24 | 0.0560 | 2 | 2 | 5 | -1.22 | -1.23 | -1.49 |
| step143k | gender | 608 | 0.0778 | 0 | 0 | 5 | +0.11 | -0.21 | -0.35 |
| step143k | profession | 496 | 0.0577 | 0 | 1 | 5 | +0.47 | +0.21 | -0.06 |
| step143k | race | 886 | 0.1118 | 1 | 1 | 3 | +0.27 | -0.00 | -0.06 |
| step143k | religion | 24 | 0.0697 | 2 | 5 | 1 | -0.93 | -0.98 | -1.21 |

---

## Normalized Indirect Effect (NIE) by layer — States (full restore)

NIE = (restoration_score - low_score) / (high_score - low_score).

- **NIE > 0**: restoring this (subject token, layer) recovers some of the clean prediction.
- **NIE = 1**: full recovery to clean-run probability.
- **NIE < 0**: restoring this position makes prediction *worse* than the corrupted baseline — the model's internal state has become inconsistent from partially restoring only one position.

⚠ Rows marked `[low-signal]` have gap < 0.03 — too small for reliable NIE estimates.

### step0  `step0`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender  ⚠ low-signal | -0.62 | -0.71 | -0.78 | -0.77 | -0.81 | -0.85 | -0.91 | -0.90 | -0.91 | -0.92 | -0.96 | -0.92 | -0.95 | -0.97 | -0.95 | -0.95 |
| profession  ⚠ low-signal | -0.44 | -0.56 | -0.62 | -0.68 | -0.71 | -0.68 | -0.69 | -0.68 | -0.75 | -0.73 | -0.78 | -0.84 | -0.89 | -0.91 | -0.90 | -0.91 |
| race  ⚠ low-signal | -0.19 | -0.25 | -0.32 | -0.34 | -0.31 | -0.31 | -0.33 | -0.37 | -0.38 | -0.41 | -0.44 | -0.47 | -0.49 | -0.53 | -0.56 | -0.56 |
| religion  ⚠ low-signal | +0.53 | +0.38 | +0.28 | +0.23 | +0.15 | +0.20 | +0.12 | +0.12 | +0.10 | +0.14 | +0.13 | +0.14 | +0.07 | +0.11 | +0.09 | +0.08 |

### step1k  `step1000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender  ⚠ low-signal | -1.12 | -1.23 | -1.24 | -1.33 | -1.42 | -1.45 | -1.49 | -1.55 | -1.56 | -1.60 | -1.59 | -1.60 | -1.57 | -1.60 | -1.62 | -1.59 |
| profession  ⚠ low-signal | -0.11 | -0.18 | -0.23 | -0.44 | -0.57 | -0.59 | -0.61 | -0.67 | -0.71 | -0.66 | -0.66 | -0.66 | -0.64 | -0.66 | -0.66 | -0.67 |
| race  ⚠ low-signal | -0.12 | -0.17 | -0.22 | -0.25 | -0.24 | -0.26 | -0.27 | -0.31 | -0.31 | -0.34 | -0.37 | -0.37 | -0.37 | -0.38 | -0.39 | -0.42 |
| religion  ⚠ low-signal | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |

### step5k  `step5000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender  ⚠ low-signal | -0.37 | -0.35 | -0.37 | -0.55 | -0.57 | -0.57 | -0.58 | -0.60 | -0.64 | -0.83 | -0.83 | -0.85 | -0.84 | -0.85 | -0.86 | -0.84 |
| profession | +0.40 | +0.34 | +0.31 | +0.30 | +0.28 | +0.28 | +0.26 | +0.25 | +0.24 | +0.02 | +0.02 | -0.13 | -0.14 | -0.17 | -0.17 | -0.18 |
| race | +0.07 | +0.01 | +0.01 | -0.05 | -0.06 | -0.08 | -0.15 | -0.15 | -0.21 | -0.23 | -0.23 | -0.27 | -0.26 | -0.27 | -0.28 | -0.27 |
| religion | -1.75 | -1.79 | -1.75 | -1.79 | -1.80 | -1.82 | -1.84 | -1.84 | -1.89 | -1.94 | -1.90 | -1.95 | -1.96 | -1.98 | -1.98 | -1.97 |

### step81k  `step81000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.09 | +0.09 | +0.06 | -0.01 | -0.08 | -0.14 | -0.20 | -0.21 | -0.24 | -0.36 | -0.36 | -0.38 | -0.36 | -0.38 | -0.38 | -0.41 |
| profession | +0.42 | +0.44 | +0.40 | +0.31 | +0.28 | +0.23 | +0.16 | +0.14 | +0.10 | -0.06 | -0.05 | -0.13 | -0.14 | -0.14 | -0.14 | -0.15 |
| race | +0.29 | +0.28 | +0.26 | +0.20 | +0.06 | +0.04 | -0.01 | -0.02 | -0.02 | -0.05 | -0.04 | -0.06 | -0.04 | -0.05 | -0.05 | -0.07 |
| religion | -0.87 | -0.75 | -0.79 | -0.91 | -0.70 | -0.73 | -0.78 | -0.78 | -0.86 | -0.92 | -0.88 | -0.96 | -0.96 | -1.01 | -1.04 | -1.08 |

### step137k  `step137000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.12 | +0.09 | +0.08 | +0.02 | -0.08 | -0.16 | -0.20 | -0.20 | -0.22 | -0.30 | -0.29 | -0.30 | -0.27 | -0.30 | -0.31 | -0.36 |
| profession | +0.50 | +0.48 | +0.45 | +0.40 | +0.35 | +0.33 | +0.26 | +0.24 | +0.24 | +0.10 | +0.11 | +0.00 | -0.00 | -0.01 | -0.02 | -0.04 |
| race | +0.28 | +0.29 | +0.26 | +0.19 | +0.02 | +0.02 | -0.01 | -0.01 | -0.00 | -0.02 | +0.00 | -0.02 | +0.01 | -0.02 | -0.02 | -0.05 |
| religion | -1.22 | -1.07 | -0.94 | -1.09 | -0.96 | -1.01 | -1.11 | -1.20 | -1.23 | -1.28 | -1.19 | -1.28 | -1.31 | -1.27 | -1.37 | -1.49 |

### step143k  `step143000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.11 | +0.08 | +0.07 | +0.03 | -0.08 | -0.14 | -0.19 | -0.20 | -0.21 | -0.28 | -0.27 | -0.27 | -0.25 | -0.29 | -0.30 | -0.35 |
| profession | +0.47 | +0.43 | +0.41 | +0.37 | +0.33 | +0.31 | +0.24 | +0.22 | +0.21 | +0.08 | +0.08 | -0.02 | -0.03 | -0.03 | -0.05 | -0.06 |
| race | +0.27 | +0.29 | +0.26 | +0.18 | +0.02 | +0.01 | -0.01 | -0.01 | -0.00 | -0.02 | -0.00 | -0.02 | +0.00 | -0.02 | -0.03 | -0.06 |
| religion | -0.93 | -0.88 | -0.77 | -0.90 | -0.88 | -0.89 | -0.94 | -0.96 | -0.98 | -1.01 | -0.95 | -1.02 | -1.07 | -1.04 | -1.10 | -1.21 |

---

## Output files

```
plots/pythia-1b/
├── stats.json                          ← full numeric data (reload without re-running)
├── report.md                           ← this file
├── heatmap_checkpoint_layer.pdf        ← checkpoint × layer heatmap (MLP + Attn)
├── {domain}-states-all-checkpoints.pdf ← all checkpoints in one figure (per domain)
├── {domain}-words-all-checkpoints.pdf
└── {label}/                            ← one folder per checkpoint
    ├── {domain}-states.pdf
    ├── {domain}-words.pdf
    ├── composite-states.pdf
    ├── composite-words.pdf
    └── composite-all.pdf
```
