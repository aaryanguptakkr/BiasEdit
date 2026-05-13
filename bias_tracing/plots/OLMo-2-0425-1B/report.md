# OLMo-2-0425-1B — Bias Tracing Report

Generated: 2026-05-12  
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
| 0B | gender | 681 | 0.0566 | 1 | 5 | 5 | -0.03 | -0.05 | -0.07 |
| 0B | profession | 548 | 0.0546 | 0 | 3 | 3 | +0.06 | -0.04 | -0.07 |
| 0B | race | 989 | 0.0735 | 0 | 5 | 4 | +0.07 | -0.01 | -0.04 |
| 0B | religion | 44 | 0.0746 | 1 | 7 | 4 | -0.02 | -0.08 | -0.12 |
| 21B | gender ⚠ | 681 | 0.0289 | 0 | 2 | 5 | -0.17 | -0.39 | -0.72 |
| 21B | profession | 548 | 0.0341 | 0 | 5 | 3 | +0.37 | +0.17 | -0.20 |
| 21B | race | 989 | 0.0456 | 2 | 4 | 4 | +0.14 | -0.01 | -0.31 |
| 21B | religion ⚠ | 44 | 0.0197 | 0 | 9 | 4 | -1.39 | -1.63 | -1.75 |
| 315B | gender | 681 | 0.0504 | 0 | 6 | 10 | +0.16 | -0.12 | -0.33 |
| 315B | profession | 548 | 0.0571 | 2 | 7 | 7 | +0.44 | +0.18 | -0.14 |
| 315B | race | 989 | 0.0803 | 2 | 6 | 2 | +0.15 | -0.04 | -0.19 |
| 315B | religion | 44 | 0.0397 | 3 | 9 | 1 | -0.65 | -0.92 | -0.87 |
| 2.4T | gender | 681 | 0.0675 | 0 | 8 | 12 | +0.22 | -0.08 | -0.20 |
| 2.4T | profession | 548 | 0.0560 | 0 | 8 | 7 | +0.44 | +0.08 | -0.10 |
| 2.4T | race | 989 | 0.0843 | 0 | 5 | 14 | +0.19 | -0.10 | -0.21 |
| 2.4T | religion | 44 | 0.0325 | 0 | 11 | 3 | -0.71 | -1.07 | -1.12 |
| 4T | gender | 681 | 0.0586 | 0 | 0 | 13 | +0.13 | -0.16 | -0.30 |
| 4T | profession | 548 | 0.0684 | 0 | 8 | 7 | +0.48 | +0.19 | -0.02 |
| 4T | race | 989 | 0.0878 | 0 | 4 | 0 | +0.24 | -0.06 | -0.17 |
| 4T | religion ⚠ | 44 | 0.0252 | 15 | 8 | 1 | -1.72 | -1.81 | -1.44 |
| s2-3B | gender | 681 | 0.0755 | 0 | 4 | 15 | +0.22 | -0.09 | -0.21 |
| s2-3B | profession | 548 | 0.0925 | 0 | 6 | 5 | +0.54 | +0.23 | +0.01 |
| s2-3B | race | 989 | 0.0849 | 0 | 4 | 0 | +0.18 | -0.08 | -0.19 |
| s2-3B | religion ⚠ | 44 | 0.0170 | 15 | 9 | 1 | -2.63 | -2.68 | -2.02 |
| s2-24B | gender | 681 | 0.0737 | 0 | 0 | 15 | +0.25 | -0.11 | -0.23 |
| s2-24B | profession | 548 | 0.0885 | 0 | 5 | 7 | +0.52 | +0.22 | +0.01 |
| s2-24B | race | 989 | 0.0948 | 0 | 4 | 14 | +0.23 | -0.05 | -0.14 |
| s2-24B | religion ⚠ | 44 | 0.0056 | 15 | 2 | 15 | -9.07 | -9.90 | -6.97 |
| s2-51B | gender | 681 | 0.0752 | 0 | 4 | 10 | +0.25 | -0.14 | -0.25 |
| s2-51B | profession | 548 | 0.0966 | 0 | 5 | 7 | +0.52 | +0.20 | +0.01 |
| s2-51B | race | 989 | 0.0941 | 0 | 4 | 2 | +0.22 | -0.06 | -0.15 |
| s2-51B | religion ⚠ | 44 | 0.0142 | 15 | 1 | 15 | -3.40 | -4.12 | -2.63 |

---

## Normalized Indirect Effect (NIE) by layer — States (full restore)

NIE = (restoration_score - low_score) / (high_score - low_score).

- **NIE > 0**: restoring this (subject token, layer) recovers some of the clean prediction.
- **NIE = 1**: full recovery to clean-run probability.
- **NIE < 0**: restoring this position makes prediction *worse* than the corrupted baseline — the model's internal state has become inconsistent from partially restoring only one position.

⚠ Rows marked `[low-signal]` have gap < 0.03 — too small for reliable NIE estimates.

### 0B  `stage1-step0-tokens0B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | -0.03 | -0.02 | -0.04 | -0.05 | -0.05 | -0.06 | -0.06 | -0.06 | -0.05 | -0.06 | -0.06 | -0.07 | -0.07 | -0.07 | -0.07 | -0.07 |
| profession | +0.06 | +0.03 | +0.02 | -0.01 | -0.01 | -0.01 | -0.01 | -0.04 | -0.04 | -0.04 | -0.05 | -0.05 | -0.06 | -0.06 | -0.06 | -0.07 |
| race | +0.07 | +0.05 | +0.04 | +0.02 | +0.01 | +0.01 | -0.00 | -0.01 | -0.01 | -0.01 | -0.02 | -0.02 | -0.02 | -0.02 | -0.03 | -0.04 |
| religion | -0.02 | +0.02 | +0.00 | -0.03 | -0.04 | -0.03 | -0.06 | -0.08 | -0.08 | -0.10 | -0.09 | -0.10 | -0.10 | -0.10 | -0.12 | -0.12 |

### 21B  `stage1-step10000-tokens21B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender  ⚠ low-signal | -0.17 | -0.19 | -0.21 | -0.22 | -0.24 | -0.27 | -0.28 | -0.32 | -0.39 | -0.44 | -0.44 | -0.50 | -0.65 | -0.66 | -0.70 | -0.72 |
| profession | +0.37 | +0.36 | +0.35 | +0.33 | +0.32 | +0.29 | +0.28 | +0.22 | +0.17 | +0.13 | +0.14 | +0.04 | -0.06 | -0.14 | -0.18 | -0.20 |
| race | +0.14 | +0.16 | +0.16 | +0.14 | +0.13 | +0.11 | +0.11 | +0.02 | -0.01 | -0.06 | -0.08 | -0.16 | -0.25 | -0.29 | -0.31 | -0.31 |
| religion  ⚠ low-signal | -1.39 | -1.48 | -1.60 | -1.76 | -1.75 | -1.69 | -1.65 | -1.62 | -1.63 | -1.78 | -1.94 | -1.97 | -1.93 | -1.90 | -1.80 | -1.75 |

### 315B  `stage1-step150000-tokens315B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.16 | +0.12 | +0.09 | +0.08 | +0.07 | +0.02 | +0.00 | -0.09 | -0.12 | -0.18 | -0.18 | -0.26 | -0.32 | -0.34 | -0.34 | -0.33 |
| profession | +0.44 | +0.48 | +0.50 | +0.43 | +0.41 | +0.33 | +0.32 | +0.22 | +0.18 | +0.12 | +0.11 | +0.02 | -0.06 | -0.12 | -0.14 | -0.14 |
| race | +0.15 | +0.16 | +0.16 | +0.13 | +0.11 | +0.04 | +0.04 | -0.03 | -0.04 | -0.08 | -0.09 | -0.13 | -0.16 | -0.19 | -0.19 | -0.19 |
| religion | -0.65 | -0.85 | -0.76 | -0.64 | -0.71 | -0.77 | -0.79 | -0.82 | -0.92 | -0.91 | -0.94 | -0.88 | -0.94 | -0.94 | -0.95 | -0.87 |

### 2.4T  `stage1-step1140000-tokens2391B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.22 | +0.17 | +0.13 | +0.08 | +0.06 | -0.01 | -0.02 | -0.08 | -0.08 | -0.13 | -0.13 | -0.18 | -0.20 | -0.21 | -0.20 | -0.20 |
| profession | +0.44 | +0.38 | +0.33 | +0.25 | +0.24 | +0.15 | +0.15 | +0.09 | +0.08 | +0.01 | -0.01 | -0.09 | -0.11 | -0.13 | -0.11 | -0.10 |
| race | +0.19 | +0.16 | +0.14 | +0.09 | +0.08 | -0.01 | -0.02 | -0.09 | -0.10 | -0.15 | -0.15 | -0.17 | -0.20 | -0.21 | -0.21 | -0.21 |
| religion | -0.71 | -0.84 | -0.99 | -0.89 | -0.95 | -0.96 | -0.93 | -1.02 | -1.07 | -1.14 | -1.19 | -1.32 | -1.35 | -1.27 | -1.26 | -1.12 |

### 4T  `stage1-step1907359-tokens4001B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.13 | +0.10 | +0.09 | +0.04 | +0.04 | -0.06 | -0.08 | -0.15 | -0.16 | -0.21 | -0.23 | -0.27 | -0.30 | -0.31 | -0.30 | -0.30 |
| profession | +0.48 | +0.45 | +0.44 | +0.36 | +0.36 | +0.27 | +0.25 | +0.20 | +0.19 | +0.13 | +0.12 | +0.05 | +0.01 | -0.01 | -0.01 | -0.02 |
| race | +0.24 | +0.20 | +0.18 | +0.13 | +0.12 | +0.03 | +0.03 | -0.05 | -0.06 | -0.11 | -0.11 | -0.14 | -0.17 | -0.17 | -0.17 | -0.17 |
| religion  ⚠ low-signal | -1.72 | -1.66 | -1.85 | -1.73 | -1.69 | -1.89 | -1.82 | -1.77 | -1.81 | -1.79 | -1.83 | -1.82 | -1.76 | -1.63 | -1.60 | -1.44 |

### s2-3B  `stage2-ingredient3-step1000-tokens3B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.22 | +0.20 | +0.16 | +0.10 | +0.10 | +0.02 | +0.00 | -0.08 | -0.09 | -0.14 | -0.15 | -0.20 | -0.22 | -0.22 | -0.22 | -0.21 |
| profession | +0.54 | +0.51 | +0.49 | +0.43 | +0.42 | +0.34 | +0.32 | +0.24 | +0.23 | +0.15 | +0.13 | +0.07 | +0.04 | +0.02 | +0.01 | +0.01 |
| race | +0.18 | +0.15 | +0.14 | +0.10 | +0.11 | +0.04 | +0.03 | -0.06 | -0.08 | -0.12 | -0.12 | -0.15 | -0.18 | -0.19 | -0.19 | -0.19 |
| religion  ⚠ low-signal | -2.63 | -2.41 | -2.75 | -2.42 | -2.24 | -2.64 | -2.65 | -2.65 | -2.68 | -2.68 | -2.68 | -2.77 | -2.66 | -2.48 | -2.36 | -2.02 |

### s2-24B  `stage2-ingredient3-step11000-tokens24B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.25 | +0.21 | +0.19 | +0.13 | +0.12 | +0.01 | -0.01 | -0.10 | -0.11 | -0.16 | -0.17 | -0.22 | -0.23 | -0.24 | -0.24 | -0.23 |
| profession | +0.52 | +0.49 | +0.47 | +0.41 | +0.40 | +0.31 | +0.28 | +0.23 | +0.22 | +0.16 | +0.14 | +0.07 | +0.04 | +0.02 | +0.01 | +0.01 |
| race | +0.23 | +0.19 | +0.17 | +0.14 | +0.13 | +0.05 | +0.05 | -0.03 | -0.05 | -0.08 | -0.08 | -0.11 | -0.14 | -0.14 | -0.14 | -0.14 |
| religion  ⚠ low-signal | -9.07 | -8.91 | -9.65 | -9.58 | -8.81 | -9.49 | -9.26 | -9.64 | -9.90 | -9.98 | -9.56 | -9.15 | -9.38 | -8.61 | -8.01 | -6.97 |

### s2-51B  `stage2-ingredient3-step23852-tokens51B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.25 | +0.18 | +0.15 | +0.10 | +0.10 | -0.01 | -0.04 | -0.14 | -0.14 | -0.18 | -0.18 | -0.23 | -0.25 | -0.26 | -0.26 | -0.25 |
| profession | +0.52 | +0.49 | +0.46 | +0.40 | +0.39 | +0.29 | +0.27 | +0.21 | +0.20 | +0.14 | +0.13 | +0.07 | +0.04 | +0.02 | +0.00 | +0.01 |
| race | +0.22 | +0.18 | +0.17 | +0.14 | +0.14 | +0.06 | +0.05 | -0.04 | -0.06 | -0.09 | -0.09 | -0.12 | -0.14 | -0.15 | -0.14 | -0.15 |
| religion  ⚠ low-signal | -3.40 | -3.29 | -3.39 | -3.30 | -3.15 | -3.62 | -3.64 | -4.05 | -4.12 | -3.86 | -3.63 | -3.38 | -3.45 | -3.26 | -3.03 | -2.63 |

---

## Output files

```
plots/OLMo-2-0425-1B/
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
