# OLMo-2-0425-1B-Instruct — Bias Tracing Report

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
| step200 | gender | 687 | 0.0788 | 0 | 2 | 1 | +0.69 | +0.36 | +0.05 |
| step200 | profession | 562 | 0.1182 | 0 | 6 | 3 | +0.70 | +0.48 | +0.09 |
| step200 | race | 1043 | 0.1150 | 0 | 5 | 1 | +0.66 | +0.46 | +0.03 |
| step200 | religion | — | — | — | — | — | — | — | — |
| step1400 | gender | 687 | 0.0803 | 0 | 2 | 1 | +0.69 | +0.35 | +0.05 |
| step1400 | profession | 562 | 0.1194 | 0 | 6 | 4 | +0.70 | +0.47 | +0.08 |
| step1400 | race | 1043 | 0.1160 | 0 | 5 | 0 | +0.66 | +0.46 | +0.03 |
| step1400 | religion | — | — | — | — | — | — | — | — |
| step2000 | gender | 687 | 0.0799 | 0 | 2 | 1 | +0.69 | +0.36 | +0.05 |
| step2000 | profession | 562 | 0.1198 | 0 | 6 | 3 | +0.70 | +0.47 | +0.08 |
| step2000 | race | 1043 | 0.1166 | 0 | 6 | 0 | +0.66 | +0.46 | +0.03 |
| step2000 | religion | — | — | — | — | — | — | — | — |
| step2600 | gender | 687 | 0.0803 | 0 | 2 | 1 | +0.69 | +0.36 | +0.05 |
| step2600 | profession | 562 | 0.1186 | 0 | 6 | 3 | +0.71 | +0.48 | +0.08 |
| step2600 | race | 1043 | 0.1169 | 0 | 5 | 1 | +0.65 | +0.46 | +0.03 |
| step2600 | religion | — | — | — | — | — | — | — | — |

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

### step200  `step_200`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.69 | +0.67 | +0.66 | +0.63 | +0.60 | +0.50 | +0.48 | +0.37 | +0.36 | +0.26 | +0.24 | +0.17 | +0.11 | +0.09 | +0.06 | +0.05 |
| profession | +0.70 | +0.69 | +0.69 | +0.66 | +0.64 | +0.58 | +0.56 | +0.48 | +0.48 | +0.36 | +0.33 | +0.24 | +0.19 | +0.16 | +0.10 | +0.09 |
| race | +0.66 | +0.69 | +0.68 | +0.68 | +0.69 | +0.61 | +0.59 | +0.50 | +0.46 | +0.37 | +0.32 | +0.25 | +0.13 | +0.06 | +0.06 | +0.03 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### step1400  `step_1400`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.69 | +0.66 | +0.66 | +0.62 | +0.60 | +0.49 | +0.48 | +0.36 | +0.35 | +0.25 | +0.23 | +0.17 | +0.11 | +0.09 | +0.06 | +0.05 |
| profession | +0.70 | +0.68 | +0.69 | +0.66 | +0.64 | +0.57 | +0.56 | +0.48 | +0.47 | +0.36 | +0.33 | +0.24 | +0.19 | +0.16 | +0.10 | +0.08 |
| race | +0.66 | +0.69 | +0.68 | +0.68 | +0.69 | +0.61 | +0.59 | +0.50 | +0.46 | +0.37 | +0.32 | +0.25 | +0.13 | +0.06 | +0.06 | +0.03 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### step2000  `step_2000`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.69 | +0.66 | +0.66 | +0.62 | +0.60 | +0.50 | +0.48 | +0.36 | +0.36 | +0.26 | +0.23 | +0.17 | +0.11 | +0.09 | +0.06 | +0.05 |
| profession | +0.70 | +0.68 | +0.68 | +0.65 | +0.63 | +0.57 | +0.55 | +0.47 | +0.47 | +0.36 | +0.33 | +0.24 | +0.19 | +0.16 | +0.10 | +0.08 |
| race | +0.66 | +0.69 | +0.68 | +0.68 | +0.69 | +0.61 | +0.59 | +0.50 | +0.46 | +0.36 | +0.32 | +0.25 | +0.13 | +0.06 | +0.06 | +0.03 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### step2600  `step_2600`

| Domain | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender | +0.69 | +0.66 | +0.66 | +0.62 | +0.60 | +0.50 | +0.48 | +0.36 | +0.36 | +0.26 | +0.23 | +0.17 | +0.11 | +0.09 | +0.06 | +0.05 |
| profession | +0.71 | +0.69 | +0.69 | +0.66 | +0.64 | +0.58 | +0.56 | +0.48 | +0.48 | +0.36 | +0.33 | +0.24 | +0.19 | +0.16 | +0.10 | +0.08 |
| race | +0.65 | +0.69 | +0.67 | +0.67 | +0.69 | +0.61 | +0.59 | +0.49 | +0.46 | +0.36 | +0.32 | +0.25 | +0.13 | +0.06 | +0.06 | +0.03 |
| religion | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

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
