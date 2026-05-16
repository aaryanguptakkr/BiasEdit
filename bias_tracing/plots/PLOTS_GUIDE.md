# Bias Tracing — Plots Guide

---

## Repository structure

```
bias_tracing/
├── fig.py                          main plotting script (within-model + cross-patch)
├── plot_utils.py                   shared constants, colors, helpers — single source of truth
├── generate_bias_plots.sh          shell pipeline: bars → delta → compare
│
├── experiments/
│   ├── bias_trace.py               causal tracing experiment (generates .npz files)
│   └── sanity_check.py
│
├── dsets/
│   └── stereoset.py                StereoSet dataset loader (sentence pairs)
│
├── scripts/
│   ├── regenerate_compare_plots.py cross-model comparison plots (reads stats.json)
│   ├── plot_checkpoint_heatmap.py  checkpoint × layer heatmap
│   ├── regenerate_reports.py       regenerate report.md from stats.json
│   └── build_html_report.py        HTML report builder
│
├── rome/                           ROME implementation (reference; not used for tracing)
├── util/                           nethook, generate, logit_lens, runningstats
├── data/                           domain-level data files
├── results/                        local extracted NPZ cache (optional; zip is canonical)
├── logs/                           per-model/domain run logs
└── plots/                          all output plots  ← you are here
    ├── PLOTS_GUIDE.md
    ├── OLMo-2-0425-1B/
    ├── OLMo-2-0425-1B-Instruct/
    ├── pythia-1b/
    ├── compare/
    └── cross_patch/
```

---

## Models

### OLMo-2-0425-1B  (`allenai/OLMo-2-0425-1B`)
- **Architecture:** decoder-only transformer, 1 B parameters, **16 layers**, 2 048 hidden dim
- **Role:** base pre-training model — the primary subject of analysis
- **8 checkpoints** spanning base pre-training (stage 1) and continued training (stage 2):

| Label | Checkpoint | Tokens seen |
|---|---|---|
| `0B` | stage1-step0-tokens0B | 0 |
| `21B` | stage1-step10000-tokens21B | 21 B |
| `315B` | stage1-step150000-tokens315B | 315 B |
| `2.4T` | stage1-step1140000-tokens2391B | 2.4 T |
| `4T` | stage1-step1907359-tokens4001B | 4 T |
| `s2-3B` | stage2-ingredient3-step1000-tokens3B | +3 B (stage 2) |
| `s2-24B` | stage2-ingredient3-step11000-tokens24B | +24 B |
| `s2-51B` | stage2-ingredient3-step23852-tokens51B | +51 B |

### OLMo-2-0425-1B-Instruct  (`allenai/OLMo-2-0425-1B-Instruct`)
- **Architecture:** same as base, 1 B parameters, **16 layers**
- **Role:** instruction fine-tuned from OLMo-2-0425-1B — compared against base to isolate the effect of RLHF/SFT on bias
- **3 checkpoints** from the fine-tuning run:

| Label | Checkpoint |
|---|---|
| `step200` | step_200 |
| `step1400` | step_1400 |
| `step2600` | step_2600 |

### pythia-1b  (`EleutherAI/pythia-1b`)
- **Architecture:** decoder-only transformer, 1 B parameters, **16 layers**
- **Role:** cross-architecture reference — verifies whether findings generalize beyond OLMo
- **6 checkpoints** covering early to late training:

| Label | Checkpoint |
|---|---|
| `step0` | step0 |
| `step1k` | step1000 |
| `step5k` | step5000 |
| `step81k` | step81000 |
| `step137k` | step137000 |
| `step143k` | step143000 |

---

## Layers

All three models have **16 transformer layers** (indices 0 – 15).

| Index | Name | What it represents |
|---|---|---|
| L0 | First transformer layer | Output after the first full MLP + Attn block. This is where the token embedding has been transformed once. High NIE here = bias is primarily lexical (carried in the embedding / very early representation). |
| L1 – L7 | Early–mid layers | Progressively more contextualized representations. |
| L8 | Middle layer (L15 // 2) | Used as the reference mid-layer in summary tables. |
| L9 – L14 | Late layers | High-level semantic processing. NIE typically goes negative here for bias (partial restore creates inconsistent state). |
| L15 | Last transformer layer | Final hidden state before the LM head. |

> **Note:** L0 here is `model.layers.0` — the output after the *first* transformer block, not the raw token embedding lookup. The raw embedding is not directly addressable by causal tracing.

---

## Dataset sizes

| Domain | Sentence pairs | Note |
|---|---|---|
| Gender | 681 | Reliable |
| Race | 989 | Reliable |
| Profession | 548 | Reliable |
| Religion | 44 | Too few — all ⚠ results should be treated with caution |

Source: **StereoSet** (intersentence, subject-corruption setup). Each pair: one stereotyped and one anti-stereotyped completion sharing the same template and subject tokens.

---

## What the scores measure

Three runs per sentence pair:
1. **Clean run** — normal forward pass. Records `high_score`: abs log prob diff (stereo − anti).
2. **Corrupted run** — subject token embeddings replaced with Gaussian noise. Records `low_score`: diff drops because the model can no longer use subject identity.
3. **Patched run** — same corrupted input, but one hidden state at a specific (token, layer) position is restored to its clean value. The score measures how much the prediction recovers.

**Score = abs log prob diff (stereo − anti):** how much more probable the stereotyped completion is than the anti-stereotyped one. Higher = stronger bias-consistent prediction.

**Effect gap** = `mean_high − mean_low`. How much subject identity drives the prediction.
Values `< 0.03` → ⚠ low-signal: patched scores are too noisy for reliable interpretation.

**NIE at layer L** = (Patched score at L − `mean_low`) / (`mean_high` − `mean_low`).
- NIE > 0: restoring layer L recovers some clean signal — that layer causally mediates bias.
- NIE = 1: full recovery.
- NIE < 0: restoring that layer hurts — partial restore creates an inconsistent internal state.

> `states_nie`, `attn_nie`, `mlp_nie` in `stats.json` store **raw patched scores per layer**, not normalized NIE. Apply the formula above to normalize.

---

## Shared plot conventions

**Y-axis** — `Abs. log prob diff (stereo − anti)` on all bar/line plots; `Δ` prefix on delta plots.
**X-axis** — `Layer` (0-indexed, 0 – 15).
**⚠ marker + yellow background** — low-signal subplot (`effect_gap < 0.03`).

**Three restore conditions — one bar color / one line panel each:**

| Color | Label | What is restored | What stays corrupted |
|---|---|---|---|
| Blue | Effect of single state | Full layer (MLP + Attn) | — |
| Red | Effect with Attn severed | MLP output only | Attention output |
| Green | Effect with MLP severed | Attention output only | MLP output |

File suffixes: `_mlp` → red bar (MLP-only restore); `_attn` → green bar (Attn-only restore); no suffix → blue bar (full restore).

---

## How to regenerate plots

```bash
cd bias_tracing

# Step 1 — within-model bar charts + delta plots (reads zip, ~10–15 min)
python fig.py --plots bars delta

# Step 2 — cross-patch plots (reads local NFS, ~5 min)
python fig.py --plots cross_patch

# Step 3 — comparison plots (reads stats.json, ~1 min — no zip or GPU needed)
python scripts/regenerate_compare_plots.py
```

**Useful `fig.py` flags:**
```
--model   OLMo-2-0425-1B | OLMo-2-0425-1B-Instruct | pythia-1b   (default: all)
--bias    gender | profession | race | religion                    (default: all)
--plots   bars | delta | compare | cross_patch | all              (default: all)
--direction  pre_to_post | post_to_pre                            (cross_patch only; default: both)
--source  zip | local | auto                                      (within-model data; default: zip)
```

---

## Output directory structure

```
plots/
│
├── OLMo-2-0425-1B/                      Base pre-training model (16 layers, 8 checkpoints)
│   ├── stats.json                        All numeric data — reload without re-running
│   ├── report.md                         Summary tables + per-layer NIE by checkpoint
│   ├── heatmap_checkpoint_layer.pdf      Checkpoint × layer heatmap (MLP and Attn rows)
│   ├── {domain}-states-all-checkpoints.pdf   One subplot per checkpoint, states conditions
│   ├── {domain}-words-all-checkpoints.pdf    One subplot per checkpoint, word positions
│   ├── {domain}-bias-delta.pdf           Δ signal between consecutive checkpoints
│   ├── {domain}-line-checkpoints.pdf     Layer-profile lines, one per checkpoint
│   └── {label}/                          One folder per checkpoint (e.g. 0B/, 21B/, 4T/, …)
│       ├── {domain}-states.pdf
│       ├── {domain}-words.pdf
│       ├── composite-states.pdf          All 4 domains side-by-side, states
│       ├── composite-words.pdf           All 4 domains side-by-side, words
│       └── composite-all.pdf             2×4 grid: top=states, bottom=words
│
├── OLMo-2-0425-1B-Instruct/             Instruct fine-tuned model (same layout, 3 checkpoints)
│   └── {label}/                          step200/, step1400/, step2600/
│
├── pythia-1b/                            Pythia-1B reference model (same layout, 6 checkpoints)
│   └── {label}/                          step0/, step1k/, step5k/, step81k/, step137k/, step143k/
│
├── compare/                              Cross-model comparison plots
│   ├── {domain}-base-vs-instruct.pdf     Bar chart: last base checkpoint vs all instruct checkpoints
│   ├── {domain}-base-vs-instruct-lines.pdf  Line profiles: base (solid) vs instruct (dashed)
│   ├── {domain}-olmo-vs-pythia.pdf       Line profiles: OLMo base (solid blue) vs Pythia (dashed green)
│   └── {domain}-bias-trajectory.pdf      Effect gap + NIE L0 over the full training timeline
│
└── cross_patch/                          Cross-model activation patching
    ├── pre_to_post/                       Source: OLMo base  →  Target: OLMo Instruct
    │   ├── {domain}-states.pdf
    │   ├── {domain}-words.pdf
    │   ├── composite-states.pdf
    │   ├── composite-words.pdf
    │   └── composite-all.pdf
    ├── post_to_pre/                       Source: OLMo Instruct  →  Target: OLMo base (same layout)
    ├── {domain}-directions-states.pdf     pre→post vs post→pre side-by-side, fixed Y-axis
    ├── {domain}-directions-words.pdf
    ├── 4panel-{domain}-states.pdf         4-panel comparison: within-model + cross-patch (states)
    ├── 4panel-{domain}-words.pdf          4-panel comparison: within-model + cross-patch (words)
    ├── 4panel-composite-states.pdf        All 4 domains, rows=domain cols=4 panels (states)
    └── 4panel-composite-words.pdf         All 4 domains, rows=domain cols=4 panels (words)
```

---

## Plot descriptions

### Per-model (`OLMo-2-0425-1B/`, `OLMo-2-0425-1B-Instruct/`, `pythia-1b/`)

**`{domain}-states-all-checkpoints.pdf`**
*Where does bias causal signal sit across layers, and does it shift with training?*
One subplot per training checkpoint. X = layer (0–15), Y = abs log prob diff. Three bars per layer (blue/red/green restore conditions).

**`{domain}-words-all-checkpoints.pdf`**
*Which token positions — subject words vs. prediction target — carry the bias signal?*
Same layout but compares token positions rather than restore conditions:
- Blue: subject / bias-attribute word tokens
- Red: token immediately before the prediction target
- Green: prediction target tokens

**`{domain}-bias-delta.pdf`**
*At which layers does bias signal increase or decrease between consecutive training steps?*
One subplot per consecutive checkpoint pair. Y = Δ abs log prob diff (curr − prev). Zero line drawn; positive = causal effect acquired at that layer.

**`{domain}-line-checkpoints.pdf`**
*How does the full layer profile evolve across training?*
3 panels (States / Attn-only / MLP-only). One line per checkpoint; light → dark = early → late training. Dashed = low-signal checkpoint.

**`{label}/composite-all.pdf`**
2 × 4 grid: top row = states conditions, bottom row = words conditions, columns = the 4 domains.

---

### Comparison (`compare/`)

**`{domain}-base-vs-instruct.pdf`**
*Does instruction tuning change how bias is distributed across layers?*
Bar chart grid. Last base checkpoint + all instruct checkpoints as subplots; Y-axis fixed across all for direct comparison.

**`{domain}-base-vs-instruct-lines.pdf`**
*Same question shown as line profiles for easier layer-by-layer reading.*
3-panel layout. Solid blue lines = base checkpoints, dashed orange = instruct checkpoints.

**`{domain}-olmo-vs-pythia.pdf`**
*Do OLMo and Pythia show the same layer-wise bias pattern despite different architectures?*
3-panel layout. Solid blue = OLMo base checkpoints, dashed green = Pythia. Y-axis shared.

**`{domain}-bias-trajectory.pdf`**
*How does overall bias strength and the early-layer signal evolve over the full training timeline?*
X = training checkpoint (base pre-training on the left, instruct fine-tuning on the right). Vertical dashed line marks the phase boundary. Blue line = base, orange dashed = instruct.

| Panel | Y-axis | What it shows |
|---|---|---|
| 1 | Effect gap (high − low) | Overall bias strength across training |
| 2 | NIE at L0 | Fraction of signal at first transformer layer — scale-invariant across checkpoints |
| 3 | Abs. log prob diff at L0 | Same, raw (un-normalized) — sensitive to overall bias level changes |

---

### Cross-patch (`cross_patch/`)

Cross-model patching injects activations from a *source* model into a *target* model at one (token, layer) position at a time, measuring whether source representations are sufficient to drive bias predictions in the target.

| Direction | Source model | Target model | Subdirectory |
|---|---|---|---|
| `pre_to_post` | OLMo-2-0425-1B (base) | OLMo-2-0425-1B-Instruct | `cross_patch/pre_to_post/` |
| `post_to_pre` | OLMo-2-0425-1B-Instruct | OLMo-2-0425-1B (base) | `cross_patch/post_to_pre/` |

Per-direction files mirror the within-model per-checkpoint layout (`{domain}-states.pdf`, `{domain}-words.pdf`, `composite-*.pdf`).

**`{domain}-directions-states.pdf` / `{domain}-directions-words.pdf`**
Side-by-side comparison of both directions. Y-axis fixed across directions for direct comparison.

**`4panel-{domain}-states.pdf` / `4panel-{domain}-words.pdf`**
*How does bias localization in each model compare to what transfers across models?*
4 panels with a shared Y-axis:

| Panel | Content |
|---|---|
| 1 | OLMo Stage 2 last checkpoint (`s2-51B`) — within-model causal tracing |
| 2 | OLMo Instruct last checkpoint (`step2600`) — within-model causal tracing |
| 3 | Pre → Post cross-patch |
| 4 | Post → Pre cross-patch |

Panels 1–2 show how bias is stored in each model on its own; panels 3–4 show how much of that
encoding transfers when activations are injected across models. Placing them on the same Y-axis
makes the comparison direct. Methodology follows arXiv:2504.02904.

> **Checkpoint caveat:** The cross-patch runs used the HuggingFace main branch of each model.
> For instruct this equals `step2600`; for base it may not exactly match `s2-51B`. To be confirmed.

**`4panel-composite-states.pdf` / `4panel-composite-words.pdf`**
Same 4-panel layout for all 4 bias domains in one figure (rows = domains, columns = panels).
Y-axis is fixed per row (per domain) so domains are not conflated.
