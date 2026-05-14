# OLMo-2-0425-1B-Instruct — Bias Tracing Report

Generated: 2026-05-13  
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
| step200 | gender | 681 | 0.0778 | 0 | 4 | 0 | +0.21 | -0.14 | -0.24 |
| step200 | profession | 548 | 0.1186 | 0 | 5 | 7 | +0.56 | +0.17 | +0.02 |
| step200 | race | 989 | 0.1163 | 0 | 5 | 0 | +0.24 | -0.02 | -0.14 |
| step200 | religion | 44 | 0.0344 | 1 | 1 | 9 | -1.25 | -1.36 | -1.30 |
| step1400 | gender | 681 | 0.0803 | 0 | 4 | 1 | +0.23 | -0.11 | -0.22 |
| step1400 | profession | 548 | 0.1219 | 0 | 6 | 5 | +0.56 | +0.18 | +0.02 |
| step1400 | race | 989 | 0.1182 | 0 | 5 | 3 | +0.24 | -0.01 | -0.13 |
| step1400 | religion | 44 | 0.0355 | 1 | 9 | 13 | -1.20 | -1.39 | -1.13 |
| step2600 | gender | 681 | 0.0794 | 0 | 4 | 1 | +0.23 | -0.11 | -0.23 |
| step2600 | profession | 548 | 0.1197 | 0 | 4 | 7 | +0.56 | +0.18 | +0.02 |
| step2600 | race | 989 | 0.1188 | 0 | 4 | 1 | +0.24 | -0.01 | -0.13 |
| step2600 | religion | 44 | 0.0359 | 1 | 2 | 9 | -1.29 | -1.42 | -1.36 |

---

## Normalized Indirect Effect (NIE) by layer — States (full restore)

NIE = (restoration_score - low_score) / (high_score - low_score).

- **NIE > 0**: restoring this (subject token, layer) recovers some of the clean prediction.
- **NIE = 1**: full recovery to clean-run probability.
- **NIE < 0**: restoring this position makes prediction *worse* than the corrupted baseline — the model's internal state has become inconsistent from partially restoring only one position.

⚠ Rows marked `[low-signal]` have gap < 0.03 — too small for reliable NIE estimates.

### step200  `step_200`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.21 | +0.15 | +0.15 | +0.10 | +0.10 | +0.01 | -0.01 | -0.12 | -0.14 | -0.19 | -0.20 | -0.26 | -0.26 | -0.27 | -0.26 | -0.24 |
| profession | +0.56 | +0.53 | +0.52 | +0.47 | +0.46 | +0.31 | +0.29 | +0.18 | +0.17 | +0.11 | +0.09 | +0.05 | +0.01 | +0.01 | +0.01 | +0.02 |
| race | +0.24 | +0.22 | +0.21 | +0.18 | +0.18 | +0.10 | +0.08 | +0.00 | -0.02 | -0.06 | -0.08 | -0.10 | -0.14 | -0.14 | -0.14 | -0.14 |
| religion | -1.25 | -1.11 | -1.24 | -1.27 | -1.31 | -1.34 | -1.39 | -1.43 | -1.36 | -1.42 | -1.33 | -1.39 | -1.43 | -1.46 | -1.48 | -1.30 |

### step1400  `step_1400`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.23 | +0.17 | +0.17 | +0.11 | +0.11 | +0.02 | -0.00 | -0.10 | -0.11 | -0.17 | -0.17 | -0.23 | -0.25 | -0.24 | -0.24 | -0.22 |
| profession | +0.56 | +0.52 | +0.53 | +0.48 | +0.46 | +0.32 | +0.29 | +0.19 | +0.18 | +0.13 | +0.10 | +0.06 | +0.02 | +0.02 | +0.02 | +0.02 |
| race | +0.24 | +0.22 | +0.21 | +0.18 | +0.18 | +0.09 | +0.09 | +0.01 | -0.01 | -0.06 | -0.07 | -0.10 | -0.12 | -0.13 | -0.13 | -0.13 |
| religion | -1.20 | -1.04 | -1.05 | -1.13 | -1.13 | -1.24 | -1.27 | -1.26 | -1.39 | -1.25 | -1.30 | -1.28 | -1.45 | -1.34 | -1.30 | -1.13 |

### step2600  `step_2600`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.23 | +0.19 | +0.17 | +0.12 | +0.12 | +0.01 | +0.00 | -0.10 | -0.11 | -0.17 | -0.18 | -0.21 | -0.23 | -0.26 | -0.23 | -0.23 |
| profession | +0.56 | +0.53 | +0.52 | +0.48 | +0.46 | +0.32 | +0.29 | +0.19 | +0.18 | +0.12 | +0.09 | +0.06 | +0.02 | +0.02 | +0.02 | +0.02 |
| race | +0.24 | +0.22 | +0.21 | +0.18 | +0.19 | +0.09 | +0.09 | +0.00 | -0.01 | -0.06 | -0.07 | -0.10 | -0.13 | -0.14 | -0.14 | -0.13 |
| religion | -1.29 | -1.11 | -1.21 | -1.22 | -1.25 | -1.44 | -1.36 | -1.54 | -1.42 | -1.44 | -1.42 | -1.40 | -1.48 | -1.48 | -1.40 | -1.36 |

---

## Output files

```
plots/OLMo-2-0425-1B-Instruct/
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
