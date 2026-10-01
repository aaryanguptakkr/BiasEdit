# OLMo-2-0425-1B — Bias Tracing Report

Generated: 2026-09-30  

## What this report measures

Causal tracing asks: *which (subject token, layer) positions causally mediate bias?*

For each sentence pair (stereotyped vs. anti-stereotyped), the subject tokens are corrupted
with Gaussian noise. Then, one hidden state at a time is restored to its clean value.
The **indirect effect** at (token i, layer j) = how much the prediction recovers when
only that one state is restored. Reported here as NIE (normalized by the clean–corrupted gap),
computed on the **signed** stereo-minus-anti score, not its absolute value -- so a layer that
pushes the model toward the anti-stereotype is distinguishable from one that restores the
stereotype, rather than both scoring as equally "recovered". "n/a" means this result predates
the signed score and only the absolute-value chain is available.

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
| **High score** | Mean clean-run absolute whole-sentence log-prob gap |
| **Low score** | Mean corrupted-run absolute whole-sentence log-prob gap |
| **Effect gap** | High − Low, absolute chain — reduction in separation after corruption; < 0.03 = low-signal |
| **Peak All/MLP/Attn** | Layer with highest raw patched score under each restore condition |
| **NIE L0/L-mid/L-last** | Normalized indirect effect (signed chain) at layer 0 / middle / final layer |

---

## Summary table

| Checkpoint | Domain | N | Gap | Peak All | Peak MLP | Peak Attn | NIE L0 | NIE L-mid | NIE L-last |
|---|---|---|---|---|---|---|---|---|---|
| 0B | gender | 687 | 0.0556 | 1 | 2 | 2 | n/a | n/a | n/a |
| 0B | profession | 562 | 0.0541 | 0 | 2 | 2 | n/a | n/a | n/a |
| 0B | race | 1043 | 0.0735 | 0 | 2 | 2 | n/a | n/a | n/a |
| 0B | religion | — | — | — | — | — | — | — | — |
| 21B | gender ⚠ | 687 | 0.0266 | 0 | 1 | 1 | n/a | n/a | n/a |
| 21B | profession | 562 | 0.0341 | 0 | 2 | 2 | n/a | n/a | n/a |
| 21B | race | 1043 | 0.0440 | 1 | 2 | 2 | +0.54 | +0.45 | +0.01 |
| 21B | religion | — | — | — | — | — | — | — | — |
| 315B | gender | 687 | 0.0490 | 0 | 3 | 12 | +0.65 | +0.48 | +0.04 |
| 315B | profession | 562 | 0.0569 | 2 | 7 | 9 | +0.77 | +0.56 | +0.03 |
| 315B | race | 1043 | 0.0783 | 0 | 3 | 0 | +0.51 | +0.53 | +0.02 |
| 315B | religion | — | — | — | — | — | — | — | — |
| 2.4T | gender | 687 | 0.0648 | 0 | 11 | 11 | +0.63 | +0.35 | +0.05 |
| 2.4T | profession | 562 | 0.0555 | 0 | 6 | 9 | +0.77 | +0.53 | +0.06 |
| 2.4T | race | 1043 | 0.0818 | 0 | 2 | 0 | +0.58 | +0.43 | +0.01 |
| 2.4T | religion | — | — | — | — | — | — | — | — |
| 4T | gender | 687 | 0.0573 | 0 | 2 | 10 | +0.62 | +0.32 | +0.04 |
| 4T | profession | 562 | 0.0677 | 0 | 5 | 2 | +0.74 | +0.46 | +0.08 |
| 4T | race | 1043 | 0.0850 | 0 | 2 | 1 | +0.57 | +0.39 | +0.01 |
| 4T | religion | — | — | — | — | — | — | — | — |
| s2-3B | gender | 687 | 0.0742 | 0 | 2 | 1 | +0.67 | +0.35 | +0.04 |
| s2-3B | profession | 562 | 0.0914 | 0 | 3 | 2 | +0.74 | +0.46 | +0.09 |
| s2-3B | race | 1043 | 0.0840 | 0 | 5 | 1 | +0.55 | +0.40 | +0.02 |
| s2-3B | religion | — | — | — | — | — | — | — | — |
| s2-24B | gender | 687 | 0.0690 | 0 | 2 | 1 | +0.70 | +0.36 | +0.03 |
| s2-24B | profession | 562 | 0.0876 | 0 | 2 | 3 | +0.76 | +0.45 | +0.08 |
| s2-24B | race | 1043 | 0.0928 | 0 | 2 | 1 | +0.55 | +0.38 | +0.02 |
| s2-24B | religion | — | — | — | — | — | — | — | — |
| s2-51B | gender | 687 | 0.0721 | 0 | 2 | 1 | +0.70 | +0.36 | +0.05 |
| s2-51B | profession | 562 | 0.0966 | 0 | 3 | 4 | +0.76 | +0.49 | +0.10 |
| s2-51B | race | 1043 | 0.0927 | 0 | 5 | 1 | +0.58 | +0.40 | +0.02 |
| s2-51B | religion | — | — | — | — | — | — | — | — |
| main | gender | 687 | 0.0659 | 0 | 15 | 9 | +0.72 | +0.35 | +0.05 |
| main | profession | 562 | 0.0757 | 1 | 12 | 9 | +0.76 | +0.47 | +0.09 |
| main | race | 1043 | 0.0835 | 0 | 4 | 14 | +0.65 | +0.45 | +0.02 |
| main | religion | — | — | — | — | — | — | — | — |

---

## Normalized Indirect Effect (NIE) by layer — States (full restore)

NIE = (restoration_score - low_score) / (high_score - low_score), all three on the
**signed** chain (stereo-minus-anti, not its absolute value).

- **NIE > 0**: restoring this (subject token, layer) recovers some of the clean model's own
  preference (whichever direction that preference happened to be).
- **NIE = 1**: full recovery of that preference.
- **NIE < 0**: restoring this position pushes the prediction *away* from the clean model's
  own preference — either the position pulls toward the opposite (stereo vs. anti) direction,
  or the model's internal state has become inconsistent from partially restoring only one
  position. The two aren't distinguished by this number alone.

⚠ Rows marked `[low-signal]` have gap < 0.03 — too small for reliable NIE estimates.

### 0B  `stage1-step0-tokens0B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| profession | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| race | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### 21B  `stage1-step10000-tokens21B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender  ⚠ low-signal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| profession | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| race | +0.54 | +0.56 | +0.57 | +0.57 | +0.54 | +0.54 | +0.53 | +0.46 | +0.45 | +0.43 | +0.42 | +0.32 | +0.19 | +0.08 | +0.03 | +0.01 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### 315B  `stage1-step150000-tokens315B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.65 | +0.64 | +0.65 | +0.67 | +0.67 | +0.62 | +0.61 | +0.51 | +0.48 | +0.41 | +0.40 | +0.30 | +0.14 | +0.11 | +0.05 | +0.04 |
| profession | +0.77 | +0.80 | +0.79 | +0.74 | +0.72 | +0.68 | +0.67 | +0.59 | +0.56 | +0.51 | +0.51 | +0.38 | +0.22 | +0.15 | +0.06 | +0.03 |
| race | +0.51 | +0.58 | +0.62 | +0.63 | +0.64 | +0.60 | +0.59 | +0.53 | +0.53 | +0.48 | +0.45 | +0.36 | +0.23 | +0.13 | +0.08 | +0.02 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### 2.4T  `stage1-step1140000-tokens2391B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.63 | +0.60 | +0.57 | +0.53 | +0.51 | +0.44 | +0.43 | +0.36 | +0.35 | +0.27 | +0.26 | +0.19 | +0.11 | +0.09 | +0.05 | +0.05 |
| profession | +0.77 | +0.75 | +0.75 | +0.71 | +0.70 | +0.63 | +0.62 | +0.53 | +0.53 | +0.43 | +0.40 | +0.28 | +0.19 | +0.15 | +0.08 | +0.06 |
| race | +0.58 | +0.62 | +0.62 | +0.60 | +0.59 | +0.51 | +0.49 | +0.45 | +0.43 | +0.36 | +0.35 | +0.28 | +0.15 | +0.07 | +0.06 | +0.01 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### 4T  `stage1-step1907359-tokens4001B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.62 | +0.59 | +0.59 | +0.54 | +0.53 | +0.43 | +0.42 | +0.33 | +0.32 | +0.24 | +0.23 | +0.16 | +0.09 | +0.08 | +0.05 | +0.04 |
| profession | +0.74 | +0.71 | +0.71 | +0.66 | +0.65 | +0.58 | +0.57 | +0.47 | +0.46 | +0.37 | +0.35 | +0.24 | +0.18 | +0.15 | +0.10 | +0.08 |
| race | +0.57 | +0.61 | +0.61 | +0.59 | +0.59 | +0.50 | +0.48 | +0.41 | +0.39 | +0.30 | +0.30 | +0.22 | +0.12 | +0.07 | +0.05 | +0.01 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### s2-3B  `stage2-ingredient3-step1000-tokens3B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.67 | +0.65 | +0.64 | +0.59 | +0.58 | +0.48 | +0.47 | +0.36 | +0.35 | +0.26 | +0.24 | +0.17 | +0.10 | +0.08 | +0.05 | +0.04 |
| profession | +0.74 | +0.72 | +0.71 | +0.66 | +0.65 | +0.59 | +0.57 | +0.47 | +0.46 | +0.36 | +0.34 | +0.25 | +0.19 | +0.16 | +0.11 | +0.09 |
| race | +0.55 | +0.58 | +0.59 | +0.58 | +0.58 | +0.52 | +0.51 | +0.43 | +0.40 | +0.33 | +0.32 | +0.24 | +0.12 | +0.07 | +0.05 | +0.02 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### s2-24B  `stage2-ingredient3-step11000-tokens24B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.70 | +0.69 | +0.69 | +0.65 | +0.62 | +0.50 | +0.48 | +0.37 | +0.36 | +0.28 | +0.26 | +0.18 | +0.10 | +0.08 | +0.04 | +0.03 |
| profession | +0.76 | +0.71 | +0.69 | +0.66 | +0.64 | +0.56 | +0.54 | +0.46 | +0.45 | +0.37 | +0.35 | +0.25 | +0.18 | +0.15 | +0.09 | +0.08 |
| race | +0.55 | +0.57 | +0.57 | +0.55 | +0.55 | +0.48 | +0.47 | +0.40 | +0.38 | +0.33 | +0.33 | +0.26 | +0.14 | +0.07 | +0.06 | +0.02 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### s2-51B  `stage2-ingredient3-step23852-tokens51B`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.70 | +0.67 | +0.67 | +0.64 | +0.61 | +0.51 | +0.49 | +0.37 | +0.36 | +0.28 | +0.26 | +0.19 | +0.11 | +0.09 | +0.06 | +0.05 |
| profession | +0.76 | +0.72 | +0.71 | +0.69 | +0.67 | +0.60 | +0.58 | +0.50 | +0.49 | +0.40 | +0.38 | +0.28 | +0.22 | +0.18 | +0.12 | +0.10 |
| race | +0.58 | +0.61 | +0.61 | +0.59 | +0.59 | +0.51 | +0.50 | +0.41 | +0.40 | +0.34 | +0.34 | +0.26 | +0.13 | +0.07 | +0.06 | +0.02 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### main  `main`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.72 | +0.69 | +0.68 | +0.65 | +0.62 | +0.57 | +0.51 | +0.41 | +0.35 | +0.32 | +0.28 | +0.18 | +0.15 | +0.10 | +0.05 | +0.05 |
| profession | +0.76 | +0.73 | +0.72 | +0.68 | +0.66 | +0.63 | +0.59 | +0.53 | +0.47 | +0.44 | +0.40 | +0.30 | +0.25 | +0.20 | +0.11 | +0.09 |
| race | +0.65 | +0.65 | +0.66 | +0.65 | +0.65 | +0.61 | +0.56 | +0.47 | +0.45 | +0.41 | +0.37 | +0.24 | +0.22 | +0.10 | +0.09 | +0.02 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

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
