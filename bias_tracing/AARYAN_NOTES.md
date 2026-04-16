# Bias Tracing — Aaryan's Experiment Notes

This document records the experiments I ran on top of the original BiasEdit codebase,
the models and domains I extended it to, and a guide for reproducing the runs.

---

## What I changed

### New model support (`bias_trace.py`, `dsets/stereoset.py`)

The original code supported GPT-2, RoBERTa, BERT, LLaMA, and GPT-J.
I added:

| Model | HF ID | Architecture | Notes |
|---|---|---|---|
| OLMo-2-0425-1B | `allenai/OLMo-2-0425-1B` | OLMo-2 (SwiGLU FFN) | requires `trust_remote_code=True`; BLANK spacing fix needed |
| Pythia-1B | `EleutherAI/pythia-1b` | GPT-NeoX | works with standard causal path |

Key code changes:
- `ModelAndTokenizer`: added OLMo and Pythia loading branches (with pad-token resize and embedding init)
- `StereoSetDataset`: added `isolmo` flag for the leading-space BLANK tokenisation quirk in OLMo
- `make_inputs`: made `find_token_range` None-safe; added OLMo to the BOS-prepend branch
- `bias_trace.py`: always uses `fp16` now (was conditional on "20b" in name); added run logging to `results/run_log.jsonl`; output dir now includes `model_base/` level; fixed a variable-name bug (`inp_stereo_origin` → `inp_anti_origin`)

### New bias domains

Added `profession` and `religion` StereoSet domain files (same format as the existing `gender`/`race` files):

- `data/domain/profession.json` — 810 items
- `data/domain/religion.json` — 79 items

`fig.py` was updated to accept `profession` and `religion` as `--bias` choices, and output is now PNG (not PDF).

### New scripts

| Script | What it does |
|---|---|
| `scripts/olmo_all.sh` | Runs all 4 domains for OLMo-2-0425-1B in sequence |
| `scripts/sanity_check.sh` | Quick end-to-end sanity check (3 samples) on GPU 2 |
| `experiments/sanity_check.py` | 6-stage PASS/FAIL checker for the OLMo pipeline |

### Compute profiling docs

- `compute_profile.md` — wall-clock and s/layer/sentence measurements across models, domains, GPU contention levels
- `compute_profile_other_methods.md` — equivalent profiling for instruction-vectors (spatiotemporal) and EasySteer pipelines

---

## Experiments I ran

All runs use `StereoSet` data. Precision is `fp16` throughout (set globally in `bias_trace.py`).
Results land in `results/<model_base>/<run_dir>/`.

### pythia-1b — all 4 domains (complete)

| Domain | Items | GPU | Procs | Wall time |
|---|---|---|---|---|
| gender | 1026 | 0 | ~12 | ~4.9 hr |
| religion | 79 | 0 | ~12 | ~0.5 hr |
| profession | 810 | 6 | 4 | ~3.8 hr |
| race | 2682 | 6 | 3 | ~12.6 hr |

Run script: `scripts/olmo_all.sh` with `MODEL=EleutherAI/pythia-1b` (or invoke directly via `bias_trace.py`).

### OLMo-2-0425-1B — all 4 domains (complete)

| Domain | Items | GPU | Procs | Precision used | Wall time |
|---|---|---|---|---|---|
| gender | 1026 | 7 | 2 | fp16 | ~2.7 hr |
| religion | 79 | 2 | 6 | fp32* | ~3.8 hr |
| profession | 810 | 2 | 2 | fp32* | ~27.6 hr |
| race | 2682 | 2 | 2 | fp32* | ~32.5 hr |

*Early runs used fp32 before the global fp16 switch was committed. Rerunning with fp16 will be ~4–10× faster.

Run script: `scripts/olmo_all.sh`

---

## How to reproduce

### Setup

```bash
conda create -n bias_trace_olmo python=3.9
conda activate bias_trace_olmo
pip install -r requirements.txt
# OLMo requires trust_remote_code, which transformers >= 4.36 handles automatically
```

### Run a single domain

From the `bias_tracing/` directory:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python experiments/bias_trace.py \
    --model_name allenai/OLMo-2-0425-1B \
    --bias_file data/domain/gender.json
```

Available `--model_name` values: `gpt2`, `roberta-large`, `bert-large-cased`, `gpt-j-6b`,
`allenai/OLMo-2-0425-1B`, `EleutherAI/pythia-1b`.

Available `--bias_file` domains: `gender`, `race`, `profession`, `religion`
(all under `data/domain/<domain>.json`).

### Run all 4 domains for OLMo

```bash
cd bias_tracing
CUDA_VISIBLE_DEVICES=<gpu_id> bash scripts/olmo_all.sh
```

### Sanity check (fast, 3 samples only)

```bash
cd bias_tracing
bash scripts/sanity_check.sh
# or, to pick a different GPU:
CUDA_VISIBLE_DEVICES=<gpu_id> python experiments/sanity_check.py --n_samples 3
```

The sanity check runs 6 stages (imports → CUDA → model load → dataset → make_inputs/noise → end-to-end trace)
and reports PASS/FAIL for each. Writes to `results/sanity_check/` only — never overwrites causal trace results.

### Plot results

```bash
cd bias_tracing
python fig.py --model_name OLMo-2-0425-1B --bias gender
# or: --bias race / profession / religion
```

Plots are saved as PNG to `results/<model_name>-<bias>-states.png` and `results/<model_name>-<bias>-words.png`.

### Reading the run log

Each run appends a JSON line to `results/run_log.jsonl`:

```bash
cat results/run_log.jsonl | python -m json.tool
```

Fields: `status`, `model_name`, `domain`, `output_dir`, `num_layers`, `num_samples`, `start_time`.

---

## Performance notes

See `compute_profile.md` for full wall-time and s/layer/sentence tables across all runs.
Key takeaways:

- **fp16 is the single biggest speedup**: OLMo fp32 at 2 procs = 7.66 s/lyr/sent; OLMo fp16 at 2 procs = 0.60 s/lyr/sent (13× faster).
- **pythia-1b fp16 is nearly contention-immune**: 3 procs → 12 procs only moves it 1.05 → 1.56×.
- **OLMo is architecture-bound at fp32**: even with only 2 processes on the GPU, it runs 7× slower per layer than pythia-1b at baseline — this is the SwiGLU FFN cost, not contention.
- For EasySteer and instruction-vectors timing comparisons, see `compute_profile_other_methods.md`.
