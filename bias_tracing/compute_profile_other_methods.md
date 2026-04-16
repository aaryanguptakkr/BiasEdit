# Compute Profile: Alternative Methodologies

Companion to `compute_profile.md` (causal tracing / bias_trace.py).

Covers two other experiment pipelines that share the same bias analysis task but use fundamentally different compute structures:
1. **instruction-vectors** — spatiotemporal causal tracing via nnsight (`bias_probe.py` / `activation_patching.py`)
2. **EasySteer** — diffmean/PCA steer vectors via vLLM (`hidden_states/capture.py` + `steer.ipynb`)

---

## Why s/layer/sentence is not directly portable

The `compute_profile.md` metric was designed for the **causal tracing loop** in `bias_trace.py`:

```
for sentence in tqdm(dataset):          # ← tqdm unit = 1 sentence
    for layer L in 0..n_layers:
        for token position t in 0..n_tokens:
            run ONE forward pass          # n_tokens × n_layers forward passes per sentence
```

`s/layer/sentence = s/it ÷ n_layers` cancels the layer dimension and isolates per-layer cost.

The other two methodologies have different compute shapes:

| Methodology | Inner loop per tqdm item | Forward passes per item |
|---|---|---|
| bias_trace.py (causal tracing) | L × t (explicit nested) | n_layers × avg_tokens |
| bias_probe.py (spatiotemporal) | L × offset × 2_dirs × seeds (explicit nested) | n_layers × avg_offset × 2 × n_seeds |
| EasySteer (capture + steer) | No layer loop — all layers in one pass | 1 per sentence (vLLM batched) |

**Consequence:**
- For **instruction-vectors spatiotemporal**: the s/layer/sentence formula gives a number that cannot be compared to causal tracing because one "layer step" in the tqdm item also hides avg_offset × 2 × n_seeds forward passes. The right unit is **s/forward-pass** or total forward pass count.
- For **EasySteer**: the layer dimension is entirely absent from the tqdm loop — vLLM runs all layers in a single fused forward pass. Dividing by n_layers would be meaningless. The right unit is **s/sentence**.

Adapted metrics are defined below for each.

---

## 1. instruction-vectors: Spatiotemporal Causal Tracing

**Repo:** `arxiv2026-instruction-vectors/`
**Script:** `experiments/bias_probe.py` → `experiments/activation_patching.py::spatiotemporal_causal_tracing()`
**Run date:** 2026-04-08 (from OBSERVATORY_REPORT.md)

### What it computes

For each sentence pair (stereo, anti), for each layer L, for each token offset k from the divergence point, for each noise seed:
- 2 clean runs (cache all layer hidden states for stereo and anti)
- n_layers × n_offsets × 2 directions × n_noise_seeds patched runs

Forward pass count per pair:
```
clean_runs       = 2
patched_runs     = n_layers × n_offsets × 2 × n_noise_seeds
total_per_pair   = 2 + (n_layers × n_offsets × 2 × n_noise_seeds)
```

### Model and dataset

| Field | Value |
|---|---|
| Model | allenai/OLMo-2-0425-1B |
| Precision | default (bfloat16) |
| Layers (n_layers) | 16 |
| Framework | nnsight (no vLLM) |
| Dataset | CrowS-Pairs (English) |
| Total pairs | 1,666 |
| Kept after same-length filter | 1,146 (69%) |
| n_noise_seeds | 5 (default) |
| Mean suffix length | 7.4 tokens |
| max_position_offset | 43 (largest observed suffix, gender_race-color group) |

### Forward pass count

| Scope | n_offsets used | Forward passes |
|---|---|---|
| Per pair (mean suffix) | avg 7.4 | 2 + (16 × 7.4 × 2 × 5) = **1,186** |
| Per pair (max suffix, worst case) | 43 | 2 + (16 × 43 × 2 × 5) = **6,882** |
| Total, 1,146 pairs (mean estimate) | avg 7.4 | ~**1,359,156** |
| Total, 1,146 pairs (worst case) | 43 | ~**7,886,772** |

This is 80–500× more forward passes per sentence than causal tracing (which uses ~n_layers × avg_tokens ≈ 16 × 15 = 240 per sentence).

### Timing

**No wall time was logged.** There is no log file for this run. Only the OBSERVATORY_REPORT.md (analysis output) was saved.

**Estimated wall time** (based on OLMo-2-1B fp16 fast-condition: 0.60 s/layer/sentence from `compute_profile.md`):
- A single causal tracing forward pass through OLMo-1B fp16 ≈ s/it / (n_layers × avg_tokens) ≈ 9.56 / (16 × 15) ≈ **0.040 s/forward-pass**
- 1,146 pairs × 1,186 forward passes × 0.040 s ≈ **54,400 s ≈ 15 hrs** (mean suffix, low contention)
- 1,146 pairs × 6,882 forward passes × 0.040 s ≈ **315,000 s ≈ 87 hrs** (worst case)

These are rough estimates. nnsight adds overhead over native PyTorch, and GPU contention at the time of the run is unknown.

### Adapted metric table

| Metric | Value | Notes |
|---|---|---|
| n_pairs processed | 1,146 | after same-length filter |
| avg forward passes/pair | ~1,186 | at mean suffix 7.4 |
| max forward passes/pair | ~6,882 | at max suffix 43 |
| Total forward passes | ~1.36M (mean) – ~7.9M (worst) | large range due to suffix length variance |
| Wall time | **not logged** | no log file saved for this run |
| Estimated wall time | ~15 hrs (mean) | based on OLMo fp16 fast-condition |
| GPU | unknown | not recorded |
| Procs on GPU | unknown | not recorded |
| Contention | unknown | not recorded |
| Data quality | theoretical estimate | no empirical timing |

---

## 2. EasySteer: Diffmean / PCA Steer Vectors

**Repo:** `EasySteer/`
**Scripts:** `easysteer/hidden_states/capture.py` (data phase), `experiment/bias/steer.ipynb` (steer phase)
**Framework:** vLLM v0.13.0+easysteer (FLASH_ATTN backend, eager mode, chunked prefill)

### What it computes

Two phases:

**Phase 1 — Data (hidden state capture):**
```
for sentence in dataset:          # ← 1 forward pass per sentence via vLLM
    capture hidden states         # all 16 layers captured simultaneously in 1 pass
compute diffmean / PCA vectors    # offline, GPU-free
```

**Phase 2 — Steer (inference with injected vector):**
```
for test_sentence in dataset:     # ← 1 forward pass per sentence via vLLM
    inject steer vector at layer L
    run inference
```

Total: **~2 forward passes per sentence** (1 capture + 1 steer). Compare to causal tracing's ~240 per sentence and spatiotemporal's ~1,186 per sentence.

### OLMo-2-0425-1B run (pipeline.log)

| Field | Value |
|---|---|
| Model | allenai/OLMo-2-0425-1B |
| Precision | bfloat16 |
| Layers | 16 |
| GPU | unknown (log records host IP, not GPU id) |
| Model weight memory | 2.77 GiB |
| KV cache available | 6.49 GiB |
| Estimated memory used by others | ~39.7 GiB |
| Estimated other procs on GPU | **~14** (39.7 ÷ 2.77 GiB/proc) |
| Condition | **extreme** (GPU nearly saturated) |
| Date | 2026-03-25 |

| Phase | Wall time | Notes |
|---|---|---|
| data.ipynb | **not in log** | pipeline.log starts at [2/3] — data phase missing |
| steer.ipynb | **54m 28s** (01:25:38 → 02:20:06) | includes ~10s model init; actual inference ~54 min |
| make_plots.py | negligible | |
| Dataset size | **unknown** | n_sentences not logged |
| s/sentence | **cannot compute** | dataset size not in log |

### pythia-1b run (pipeline_pythia.log)

| Field | Value |
|---|---|
| Model | EleutherAI/pythia-1b |
| Precision | float16 |
| Layers | 16 |
| GPU | 5 |
| Model weight memory | 1.88 GiB |
| KV cache available | 35.38 GiB |
| Estimated memory used by others | ~11.7 GiB |
| Estimated other procs on GPU | **~6** (11.7 ÷ 1.88 GiB/proc) |
| Condition | **adverse** (moderate–heavy sharing) |
| Date | 2026-03-25 |

| Phase | Wall time | Notes |
|---|---|---|
| data.ipynb | **51 s** (12:48:01 → 12:48:52) | hidden state capture, all categories |
| steer.ipynb | **36m 1s** (12:49:31 → 13:25:32) | inference + evaluation |
| make_plots.py | negligible | |
| Dataset size | **unknown** | n_sentences not logged |
| s/sentence | **cannot compute** | dataset size not in log |

### Additional steer.ipynb run (run_analysis_after_steer.log)

| Field | Value |
|---|---|
| Date | 2026-03-23 |
| Model | unknown |
| steer.ipynb wall time | **29m 0s** (21:01:38 → 21:30:38 PDT) |
| analysis.ipynb wall time | **24s** |
| GPU / contention | unknown |

This run is 7 minutes faster than the pythia run (36 min) and 25 minutes faster than the OLMo run (54 min). Likely different model or lower contention.

### Adapted metric table

| Model | Precision | Phase | GPU | Procs (est.) | Wall time | Condition | Data quality |
|---|---|---|---|---|---|---|---|
| OLMo-2-0425-1B | bfloat16 | steer | unknown | ~14 | **54m 28s** | extreme | clean (from log) |
| OLMo-2-0425-1B | bfloat16 | data | unknown | ~14 | **not logged** | extreme | missing |
| pythia-1b | fp16 | data | 5 | ~6 | **51 s** | adverse | clean (from log) |
| pythia-1b | fp16 | steer | 5 | ~6 | **36m 1s** | adverse | clean (from log) |
| unknown | unknown | steer | unknown | unknown | **29m 0s** | unknown | clean (from log) |

---

## 3. Cross-Methodology Comparison

### Compute cost per sentence (forward passes)

| Methodology | Script | Forward passes/sentence | Relative cost |
|---|---|---|---|
| EasySteer (capture) | capture.py | **1** | **1× (cheapest)** |
| EasySteer (steer) | steer.ipynb | **1** | **1×** |
| Causal tracing | bias_trace.py | ~n_layers × avg_tokens ≈ **240** | **240×** |
| Spatiotemporal (mean offset) | bias_probe.py | ~n_layers × avg_offset × 2 × seeds ≈ **1,184** | **~1,184×** |
| Spatiotemporal (worst case) | bias_probe.py | ~n_layers × max_offset × 2 × seeds ≈ **6,880** | **~6,880×** |

*Estimates assume n_layers=16, avg_tokens=15, avg_offset=7.4, max_offset=43, n_seeds=5.*

### Wall time by method and model

| Methodology | Model | Precision | Total wall time | Condition | Items |
|---|---|---|---|---|---|
| EasySteer (steer) | OLMo-2-1B | bf16 | ~54m | extreme (~14 procs) | unknown |
| EasySteer (steer) | pythia-1b | fp16 | ~36m | adverse (~6 procs) | unknown |
| EasySteer (steer) | unknown | unknown | ~29m | unknown | unknown |
| EasySteer (data) | pythia-1b | fp16 | ~51s | adverse (~6 procs) | unknown |
| Causal tracing | OLMo-2-1B | fp32 | ~27.6 hr (profession, 810 items) | fast (2 procs) | 810 |
| Causal tracing | OLMo-2-1B | fp16 | ~2.7 hr (gender, 1026 items) | fast (2 procs) | 1026 |
| Causal tracing | pythia-1b | fp16 | ~12.6 hr (race, 2682 items) | low (3 procs) | 2682 |
| Spatiotemporal | OLMo-2-1B | bf16 | **not logged** | unknown | 1,146 pairs |

### Key architectural differences

| Property | EasySteer | Causal tracing | Spatiotemporal |
|---|---|---|---|
| Inference engine | vLLM (optimized) | HuggingFace + custom hooks | nnsight (proxy-based) |
| Precision | bf16 (OLMo), fp16 (pythia) | mixed (see compute_profile.md) | bf16 (default from model) |
| Layer loop explicit? | **No** — all layers in one fwdpass | **Yes** — outer loop over layers | **Yes** — inner loop over layers |
| Position loop? | No | Yes (over token positions) | Yes (over offsets from div_pos) |
| Noise seeds? | No | No | Yes (5 per pair) |
| Bidirectional? | No | No | Yes (stereo→anti and anti→stereo) |
| Dataset | Unknown (StereoSet or similar) | CrowS-Pairs by domain | CrowS-Pairs, all 9 bias types |
| Reliability flag | ✓ Not reliable per user | ← user note | ✓ Reliable |

---

## Data gaps to fill on future runs

| Method | Model | What to log | Why it matters |
|---|---|---|---|
| EasySteer | OLMo-2-1B | data.ipynb wall time | only steer phase is known |
| EasySteer | OLMo-2-1B | GPU id and n_procs | condition label missing |
| EasySteer | OLMo-2-1B | dataset size (n_sentences) | needed for s/sentence |
| EasySteer | pythia-1b | dataset size (n_sentences) | needed for s/sentence |
| EasySteer | unknown | model and GPU for 2026-03-23 run | can't attribute 29-min timing |
| Spatiotemporal | OLMo-2-1B | wall time (add tqdm timestamps to logs) | no timing data at all |
| Spatiotemporal | OLMo-2-1B | GPU id and n_procs | condition label missing |
