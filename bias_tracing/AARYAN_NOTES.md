# Bias Tracing — Pipeline Reference

This document is a guide for running causal tracing experiments on new models/checkpoints,
and for generating plots from any existing results.

---

## Overview

The pipeline has two independent stages:

| Stage | Script | Needs GPU? | Time |
|---|---|---|---|
| **1. Data generation** | `experiments/bias_trace.py` / `scripts/olmo_all.sh` | Yes | Hours per model |
| **2. Plot generation** | `generate_bias_plots.sh` | No | ~1–2 min (zip) |

Results from stage 1 are stored as `.npz` case files under `results/`.
Stage 2 reads those files and writes PDFs + stats to `plots/`.

---

## Stage 1 — Data generation (needs GPU)

### Single model / checkpoint / domain

```bash
cd bias_tracing
CUDA_VISIBLE_DEVICES=<gpu_id> python experiments/bias_trace.py \
    --model_name <model_hf_id> \
    --branch <checkpoint_branch> \
    --bias_file data/domain/<domain>.json \
    --output_dir results/<org>/<model>/<checkpoint>/<domain>/causal_trace
```

**`--model_name`** — any HuggingFace model ID. Tested with:
- `allenai/OLMo-2-0425-1B`, `allenai/OLMo-2-0425-1B-Instruct`
- `EleutherAI/pythia-1b`
- `gpt2`, `roberta-large`, `bert-large-cased`, `gpt-j-6b`

**`--branch`** — checkpoint name (used as a subdirectory label, e.g. `step10000`, `stage1-step10000-tokens21B`).
For models without checkpoints just pass the model name again or omit.

**`--bias_file`** — one of `data/domain/gender.json`, `race.json`, `profession.json`, `religion.json`.

Output lands in:
```
results/<org>/<model>/<checkpoint>/<domain>/causal_trace/
    cases/          ← .npz files, one per sentence pair (the input to the plot scripts)
    pdfs/           ← per-case causal trace PDFs (optional visualisation)
```

### Batch run across multiple checkpoints (parallel)

`scripts/olmo_all.sh` is a template for running multiple model × checkpoint pairs in parallel.
Edit the `model_branch_pairs` array at the top to specify which combinations to run:

```bash
model_branch_pairs=(
    "EleutherAI/pythia-1b step0"
    "EleutherAI/pythia-1b step1000"
    "allenai/OLMo-2-0425-1B stage1-step10000-tokens21B"
    # add more as needed
)
```

Then run:

```bash
cd bias_tracing
# GPU is pinned to CUDA_VISIBLE_DEVICES=0 inside the script — edit to change
bash scripts/olmo_all.sh
```

The script runs up to `N_PARALLEL=5` jobs at a time using background processes and `wait -n`.
Each job calls `experiments/bias_trace.py` for one (model, checkpoint, domain) combination.

### Sanity check before a long run

```bash
cd bias_tracing
bash scripts/sanity_check.sh
# or pick a GPU explicitly:
CUDA_VISIBLE_DEVICES=<gpu_id> python experiments/sanity_check.py --n_samples 3
```

Runs 6 stages (imports → CUDA → model load → dataset → make_inputs/noise → end-to-end trace)
and reports PASS/FAIL for each. Writes to `results/sanity_check/` only.

### Supported bias domains

| File | Domain | Notes |
|---|---|---|
| `data/domain/gender.json` | Gender | ~600–680 sentence pairs |
| `data/domain/race.json` | Race | ~880–990 sentence pairs |
| `data/domain/profession.json` | Profession | ~496–548 sentence pairs |
| `data/domain/religion.json` | Religion | ~24–44 sentence pairs |

All use StereoSet format. Dtype is set per model in `bias_trace.py` — OLMo base uses `float32`,
OLMo Instruct uses `bfloat16`, Pythia uses `float16`. See performance notes for speed implications.

### Output format (`.npz` case files)

Each `.npz` file represents one stereo/anti-stereo sentence pair and contains:

| Key | Shape | Description |
|---|---|---|
| `scores` | `(n_tokens, n_layers)` | Indirect effect at each (token, layer) restore point |
| `high_score` | scalar | Baseline probability (clean run) |
| `low_score` | scalar | Corrupted-only probability |
| `corrupt_range_anti` | `(n, 2)` | Token ranges of the subject (corrupted) |
| `blank_idxs_anti` | `(2,)` | Start/end of the prediction target token |
| `input_tokens_anti/stereo` | `(n_tokens,)` | Readable token strings |
| `subject` | `(k,)` | Subject word(s) |
| `kind` | scalar str | `''` = full restore, `'mlp'` = MLP only, `'attn'` = Attn only |

Three files are written per sentence pair: base, `_mlp`, `_attn`.

---

## Stage 2 — Plot generation (no GPU)

Reads `.npz` case files from `results/` (or directly from a zip archive) and writes
summary PDFs and stats to `plots/`.

### Run

```bash
cd bias_tracing
bash generate_bias_plots.sh                    # all models found in results/, all domains
bash generate_bias_plots.sh --model pythia-1b  # single model
bash generate_bias_plots.sh --bias gender      # single domain
bash generate_bias_plots.sh --source zip       # read from a zip archive (fast, default)
bash generate_bias_plots.sh --source local     # read from extracted NFS files
```

### `--source` flag

On network filesystems (NFS / deepfreeze), reading thousands of small `.npz` files individually
is slow (~22 ms per file). Packing results into a zip and reading sequentially is ~200× faster.

| `--source` | When to use |
|---|---|
| `zip` (default) | Results stored in a zip archive on network storage |
| `local` | Results already extracted on fast local disk |
| `auto` | Prefer local if the directory exists, else fall back to zip |

Set `ZIP_PATH` at the top of `fig.py` and `scripts/plot_checkpoint_heatmap.py` to point at your archive.

### Output structure

```
plots/
├── <domain>-bias-trajectory.pdf            ← 3-panel training timeline (effect gap, NIE at L0, raw L0)
├── <domain>-base-vs-instruct.pdf           ← bar charts: last base ckpt + all instruct ckpts
├── <domain>-base-vs-instruct-lines.pdf     ← line plots: base (solid) vs instruct (dashed)
├── <domain>-olmo-vs-pythia.pdf             ← cross-architecture line comparison (OLMo vs Pythia)
└── <model>/
    ├── stats.json                              ← all numeric data; reload to avoid re-running
    ├── report.md                               ← human-readable summary (peak layers, n_cases, etc.)
    ├── heatmap_checkpoint_layer.pdf            ← checkpoint × layer heatmap (MLP + Attn)
    ├── <domain>-line-checkpoints.pdf           ← layer profiles as lines, one per checkpoint
    ├── <domain>-states-all-checkpoints.pdf     ← bar charts: all checkpoints in one figure
    ├── <domain>-words-all-checkpoints.pdf
    ├── <domain>-bias-delta.pdf                 ← Δ abs. log prob diff between consecutive checkpoints
    └── <checkpoint_label>/                     ← one folder per checkpoint
        ├── <domain>-states.pdf                 ← single / attn-severed / MLP-severed bars
        ├── <domain>-words.pdf                  ← bias-word / pre-blank / blank-token bars
        ├── composite-states.pdf                ← all 4 domains side-by-side
        ├── composite-words.pdf
        └── composite-all.pdf                   ← 2×4 grid: top=states, bottom=words
```

**Regenerating comparison plots without re-running the full pipeline:**

```bash
cd bias_tracing
python scripts/regenerate_compare_plots.py
```

Reads from existing `plots/<model>/stats.json` — no GPU or zip needed. Regenerates all top-level
PDFs (bias-trajectory, base-vs-instruct, olmo-vs-pythia, line-checkpoints per model).

### Reloading stats without re-running

`plots/<model>/stats.json` stores all computed values for every checkpoint × domain:

```python
import json
stats = json.load(open('plots/OLMo-2-0425-1B/stats.json'))
for ckpt in stats['checkpoints']:
    for domain, s in ckpt['domains'].items():
        print(ckpt['label'], domain,
              'peak_layer:', s['peak_layer_states'],
              'n_cases:', s['n_cases'])
```

Fields per domain entry: `n_cases`, `num_layers`, `mean_high`, `mean_low`, `effect_gap`,
`states_nie` / `attn_nie` / `mlp_nie` / `pre_blank_nie` / `blank_nie` (lists, one value per layer),
`peak_layer_states/mlp/attn`, `top3_states/mlp/attn`.

---

## Adding a new model or checkpoint

1. Add it to `scripts/olmo_all.sh` (or run `bias_trace.py` directly) — stage 1.
2. Add it to `MODEL_CONFIGS` in `fig.py` and `scripts/plot_checkpoint_heatmap.py` — stage 2.
3. Run `bash generate_bias_plots.sh` to regenerate plots and stats.

For OLMo-family models set `trust_remote_code=True` (handled automatically in `ModelAndTokenizer`).
For models with a leading-space BLANK tokenisation quirk, pass `isolmo=True` to `StereoSetDataset`.

---

## Code changes from the original BiasEdit codebase

### New model support (`bias_trace.py`, `dsets/stereoset.py`)

| Model | HF ID | Architecture | Notes |
|---|---|---|---|
| OLMo-2-0425-1B | `allenai/OLMo-2-0425-1B` | OLMo-2 (SwiGLU FFN) | requires `trust_remote_code=True`; BLANK spacing fix |
| Pythia-1B | `EleutherAI/pythia-1b` | GPT-NeoX | works with standard causal path |

Key changes:
- `ModelAndTokenizer`: OLMo and Pythia loading branches (pad-token resize, embedding init)
- `StereoSetDataset`: `isolmo` flag for leading-space BLANK tokenisation in OLMo
- `make_inputs`: `find_token_range` None-safe; OLMo added to BOS-prepend branch
- `bias_trace.py`: dtype is model-specific — OLMo base → `float32`, OLMo Instruct → `bfloat16`, Pythia → `float16`; run logging to `results/run_log.jsonl`; output dir includes model base; fixed `inp_stereo_origin` → `inp_anti_origin` bug

### New bias domains

- `data/domain/profession.json` — 810 items
- `data/domain/religion.json` — 79 items

---

## What causal tracing is measuring — and what we found

### The method

Causal tracing isolates *where* a model processes specific information. For each sentence pair
(stereotyped vs. anti-stereotyped from StereoSet), the subject tokens are corrupted with Gaussian
noise, then one hidden state at a time is restored to its clean value. The **indirect effect**
at position (token i, layer j) measures how much the prediction recovers from that single restore.

Three `.npz` files are written per sentence pair:
- **base** — full restore (MLP + Attn)
- **`_mlp`** — only MLP output restored, Attn left corrupted ("MLP-only" contribution)
- **`_attn`** — only Attn output restored, MLP left corrupted ("Attn-only" contribution)

Scores are aggregated over **subject token positions only** (not the full sentence, which would
pick up the prediction-target token and give misleading results).

The **NIE** (Normalized Indirect Effect):

$$\text{NIE} = \frac{\text{Patched Absolute Diff} - \text{Corrupted Absolute Diff}}{\text{Clean Absolute Diff} - \text{Corrupted Absolute Diff}}$$

Where "Absolute Diff" = |log P(stereotyped) − log P(anti-stereotyped)| for the same sentence template.

- NIE > 0: this position causally contributes to the prediction
- NIE < 0: restoring this position makes things *worse* than the corrupted baseline (inconsistent state)
- NIE = 1: full recovery

In terms of stored values: `(states_nie[L] − mean_low) / effect_gap` — where `mean_low` ≈ Corrupted
baseline and `mean_high` ≈ Clean baseline, so `effect_gap = mean_high − mean_low` = Clean − Corrupted.

> **Note on field naming in `stats.json`**: The fields `states_nie`, `attn_nie`, `mlp_nie` store
> **raw abs log prob diffs** (not normalized NIE values), despite the `_nie` suffix. They are the
> per-layer patched scores averaged across sentence pairs. Apply the NIE formula above when you
> need normalized values.

---

### Plot conventions

**Bar chart labels and colors** (follows original BiasEdit paper):

| Bar color | Label | Restore condition | What's "severed" |
|---|---|---|---|
| Blue | Effect of single state | Full restore (MLP + Attn) | — |
| Red | Effect with Attn severed | MLP-only restore | Attention disabled |
| Green | Effect with MLP severed | Attn-only restore | MLP disabled |

"Severed" means that component is left in its corrupted state and does not contribute. The label
describes what is *disabled*, while the file suffix describes what is *restored* (`_mlp` = MLP-only
restore = Attn severed → red bar; `_attn` = Attn-only restore = MLP severed → green bar).

**Y-axis**: All bar charts and line plots use `Abs. log prob diff (stereo − anti)` (raw, unnormalized).
Delta plots use `Δ Abs. log prob diff (stereo − anti)`. The bias-trajectory plot uses NIE at L0
(normalized, see formula above) in its second panel alongside the raw value in its third panel.

**⚠ Low-signal marker**: Plotted when `effect_gap < 0.03`. At that threshold the model barely
distinguishes high-bias from low-bias examples, making causal tracing estimates unreliable
(small denominator → noisy NIE). Treat those checkpoints with caution.

Notable instance: **OLMo-2-0425-1B at 21B tokens, gender domain** (`effect_gap = 0.0289`).
The model recovers to normal effect gaps by 315B tokens and keeps growing through full training.

---

## Performance notes

See `compute_profile.md` for full wall-time tables. Key takeaways:

- **OLMo base runs in float32** (by code design) — 7.66 s/layer/sent at 2 procs. Switching to fp16 gives 0.60 s/layer/sent (13× faster), but the current code does not do this for OLMo base.
- **OLMo Instruct runs in bfloat16**, Pythia in float16 — both are significantly faster than OLMo base.
- **OLMo is architecture-bound at float32** — even 2 processes runs 7× slower per layer than Pythia; this is the SwiGLU FFN cost, not contention.
- **Pythia-1B is nearly contention-immune** — 3 → 12 procs moves it only 1.05 → 1.56×.
- For EasySteer and instruction-vectors comparisons, see `compute_profile_other_methods.md`.
