# pythia-1b — Bias Tracing Report

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
| step0 | gender ⚠ | 616 | 0.0042 | 0 | 2 | 2 | n/a | n/a | n/a |
| step0 | profession ⚠ | 513 | 0.0014 | 0 | 1 | 11 | n/a | n/a | n/a |
| step0 | race ⚠ | 971 | 0.0042 | 0 | 3 | 1 | n/a | n/a | n/a |
| step0 | religion | — | — | — | — | — | — | — | — |
| step1k | gender ⚠ | 616 | 0.0142 | 0 | 1 | 1 | n/a | n/a | n/a |
| step1k | profession ⚠ | 513 | 0.0094 | 0 | 1 | 0 | n/a | n/a | n/a |
| step1k | race ⚠ | 971 | 0.0043 | 1 | 0 | 1 | n/a | n/a | n/a |
| step1k | religion | — | — | — | — | — | — | — | — |
| step5k | gender | 616 | 0.0407 | 0 | 0 | 2 | n/a | n/a | n/a |
| step5k | profession | 513 | 0.0315 | 0 | 2 | 2 | +0.77 | +0.68 | +0.05 |
| step5k | race | 971 | 0.0335 | 0 | 2 | 5 | +0.47 | +0.45 | +0.03 |
| step5k | religion | — | — | — | — | — | — | — | — |
| step81k | gender | 616 | 0.0492 | 2 | 2 | 0 | +0.69 | +0.52 | +0.05 |
| step81k | profession | 513 | 0.0391 | 1 | 2 | 2 | +0.81 | +0.71 | -0.01 |
| step81k | race | 971 | 0.0701 | 1 | 2 | 2 | +0.50 | +0.43 | +0.03 |
| step81k | religion | — | — | — | — | — | — | — | — |
| step137k | gender | 616 | 0.0498 | 2 | 0 | 0 | +0.64 | +0.55 | +0.04 |
| step137k | profession | 513 | 0.0414 | 0 | 0 | 5 | +0.74 | +0.76 | -0.01 |
| step137k | race | 971 | 0.0744 | 1 | 2 | 1 | +0.44 | +0.42 | +0.02 |
| step137k | religion | — | — | — | — | — | — | — | — |
| step143k | gender | 616 | 0.0474 | 2 | 0 | 2 | +0.64 | +0.57 | +0.03 |
| step143k | profession | 513 | 0.0410 | 0 | 2 | 1 | +0.79 | +0.75 | +0.00 |
| step143k | race | 971 | 0.0706 | 1 | 2 | 1 | +0.45 | +0.42 | +0.02 |
| step143k | religion | — | — | — | — | — | — | — | — |

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

### step0  `step0`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender  ⚠ low-signal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| profession  ⚠ low-signal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| race  ⚠ low-signal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### step1k  `step1000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender  ⚠ low-signal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| profession  ⚠ low-signal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| race  ⚠ low-signal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### step5k  `step5000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| profession | +0.77 | +0.69 | +0.71 | +0.72 | +0.70 | +0.69 | +0.67 | +0.68 | +0.68 | +0.43 | +0.44 | +0.18 | +0.17 | +0.14 | +0.09 | +0.05 |
| race | +0.47 | +0.47 | +0.49 | +0.49 | +0.49 | +0.49 | +0.45 | +0.45 | +0.45 | +0.33 | +0.34 | +0.11 | +0.10 | +0.05 | +0.04 | +0.03 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### step81k  `step81000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.69 | +0.74 | +0.73 | +0.69 | +0.67 | +0.64 | +0.53 | +0.54 | +0.52 | +0.19 | +0.19 | +0.10 | +0.07 | +0.06 | +0.06 | +0.05 |
| profession | +0.81 | +0.77 | +0.72 | +0.71 | +0.75 | +0.73 | +0.68 | +0.73 | +0.71 | +0.41 | +0.44 | +0.16 | +0.07 | +0.05 | +0.02 | -0.01 |
| race | +0.50 | +0.58 | +0.59 | +0.54 | +0.52 | +0.50 | +0.40 | +0.43 | +0.43 | +0.30 | +0.32 | +0.13 | +0.08 | +0.05 | +0.04 | +0.03 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### step137k  `step137000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.64 | +0.67 | +0.67 | +0.68 | +0.66 | +0.63 | +0.53 | +0.54 | +0.55 | +0.20 | +0.21 | +0.11 | +0.06 | +0.05 | +0.05 | +0.04 |
| profession | +0.74 | +0.74 | +0.72 | +0.74 | +0.76 | +0.77 | +0.71 | +0.78 | +0.76 | +0.44 | +0.49 | +0.20 | +0.09 | +0.07 | +0.02 | -0.01 |
| race | +0.44 | +0.55 | +0.59 | +0.57 | +0.55 | +0.52 | +0.39 | +0.43 | +0.42 | +0.29 | +0.34 | +0.12 | +0.07 | +0.04 | +0.03 | +0.02 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### step143k  `step143000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.64 | +0.67 | +0.68 | +0.69 | +0.66 | +0.64 | +0.54 | +0.55 | +0.57 | +0.20 | +0.21 | +0.11 | +0.06 | +0.05 | +0.04 | +0.03 |
| profession | +0.79 | +0.76 | +0.74 | +0.74 | +0.77 | +0.78 | +0.71 | +0.77 | +0.75 | +0.43 | +0.48 | +0.21 | +0.09 | +0.07 | +0.04 | +0.00 |
| race | +0.45 | +0.55 | +0.60 | +0.56 | +0.55 | +0.51 | +0.39 | +0.43 | +0.42 | +0.29 | +0.34 | +0.12 | +0.06 | +0.04 | +0.03 | +0.02 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

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
