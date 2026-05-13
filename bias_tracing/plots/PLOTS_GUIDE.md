# Plots Guide

## Dataset sizes

| Domain | Sentence pairs |
|---|---|
| Gender | 681 |
| Race | 989 |
| Profession | 548 |
| Religion | 44 — too few for reliable estimates; treat with caution |

---

## Setup (read this first)

Each data point comes from a **sentence pair**: one stereotyped completion and one anti-stereotyped completion sharing the same template and subject (e.g. "The nurse said ___ was tired" with stereotyped/anti-stereotyped subjects).

Three runs per pair:
1. **Clean run** — normal forward pass. Records `high_score`: abs log prob diff (stereo − anti).
2. **Corrupted run** — subject token embeddings replaced with Gaussian noise. Records `low_score`: abs log prob diff drops because the model can no longer use subject identity.
3. **Patched run** — same corrupted input, but one hidden state at a specific layer is restored to its clean value. Score measures how much the prediction recovers.

**Score = abs log prob diff (stereo − anti)**: how much more probable the stereotyped completion is than the anti-stereotyped one. Higher = more bias-consistent prediction.

**Effect gap** = `mean_high − mean_low` = average (clean score − corrupted score) across all sentence pairs. Measures how much subject identity drives the prediction.

**NIE at layer L** = (Patched score at L − `mean_low`) / (`mean_high` − `mean_low`)
— fraction of the clean signal recovered by restoring only layer L.
— NIE > 0: layer L carries causal bias signal. NIE < 0: restoring it hurts. NIE = 1: full recovery.

> `states_nie`, `attn_nie`, `mlp_nie` in `stats.json` store the **raw patched scores per layer** (not normalized NIE). Apply the formula above for NIE.

---

## Shared conventions

**Y-axis** — `Abs. log prob diff (stereo − anti)` on all bar/line plots; `Δ` prefix on delta plots.  
**X-axis** — `Layer`: transformer layer index (0-indexed). L0 = first transformer layer output (`model.layers.0`), not the raw token embedding.  
**⚠** — low-signal checkpoint (`effect_gap < 0.03`): the model barely responds to subject corruption, making patched scores unreliable.

**Three restore conditions** — each gives one bar color / one line panel:

| Color | Label | What is restored | What stays corrupted |
|---|---|---|---|
| Blue | Effect of single state | Full layer (MLP + Attn) | — |
| Red | Effect with Attn severed | MLP output only | Attention output |
| Green | Effect with MLP severed | Attention output only | MLP output |

File suffixes: `_mlp` → red bar (MLP restored); `_attn` → green bar (Attn restored); no suffix → blue bar.

---

## Files

### Bar chart plots (`<model>/`)

**`{domain}-states-all-checkpoints.pdf`**  
*Where does bias causal signal sit across layers, and does this change with training?*  
One subplot per training checkpoint. X = layer, Y = abs log prob diff. Bars as above.

**`{domain}-words-all-checkpoints.pdf`**  
*Which token positions — subject words vs. prediction target — carry the bias signal?*  
Same layout, but scores are at different token positions instead of restore conditions:
- Blue: subject/bias-attribute word tokens
- Red: token immediately before the prediction target
- Green: prediction target tokens

**`{domain}-bias-delta.pdf`**  
*At which layers does the bias signal increase or decrease between consecutive training steps?*  
One subplot per consecutive checkpoint pair. Y = change in abs log prob diff (curr − prev) at each layer. Zero line drawn. Positive = more causal effect acquired.

### Line plots (`<model>/`)

**`{domain}-line-checkpoints.pdf`**  
*How does the full layer profile of each restore condition evolve across training?*  
3 panels, one line per checkpoint (light → dark = early → late training). Dashed = low-signal.
- Panel 1 **States**: full layer restore
- Panel 2 **Attn-only**: only attention output restored (MLP stays corrupted)
- Panel 3 **MLP-only**: only MLP output restored (Attn stays corrupted)

### Comparison plots (top-level `plots/`)

**`{domain}-base-vs-instruct.pdf`**  
*Does instruction tuning change how bias is distributed across layers?*  
Bar chart grid. Last base checkpoint + all instruct checkpoints as separate subplots. Y-axis fixed across all for direct comparison. Yellow background = low-signal.

**`{domain}-base-vs-instruct-lines.pdf`**  
*Same question as above, shown as line profiles for easier layer-by-layer comparison.*  
Same 3-panel line layout. Solid lines = base checkpoints, dashed = instruct.

**`{domain}-olmo-vs-pythia.pdf`**  
*Do OLMo and Pythia show the same layer-wise bias pattern despite different architectures?*  
Same 3-panel line layout. Solid blue = OLMo checkpoints, dashed green = Pythia. Y-axis shared.

**`{domain}-bias-trajectory.pdf`**  
*How does overall bias strength and the early-layer signal evolve over the full training timeline?*  
X = training checkpoint (left = base pre-training, right = instruct fine-tuning). Vertical dashed line marks the phase boundary. Blue = base, orange dashed = instruct.

| Panel | Y-axis | What it shows |
|---|---|---|
| 1 | Effect gap (high − low) | Overall bias strength across training |
| 2 | NIE at L0 | Normalized fraction of signal at first transformer layer — scale-invariant across checkpoints |
| 3 | Abs. log prob diff at L0 | Same, raw (un-normalized) — affected by overall bias scale changes |
