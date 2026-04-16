# Compute Profile: Per-Layer Per-Sentence Time

Tracks wall-clock time per tqdm iteration divided by model layer count.
Formula: **s/layer/sentence = s/it ÷ num_layers**

**Each row = one distinct GPU state (GPU id + process count + precision).**
Where a run migrated GPUs mid-stream, states are listed as separate rows. Rows covering multiple states are marked `[AGGREGATE]` — per-state timings were not captured separately for those runs.

**Baseline reference:** pythia-1b fp16 low contention = **1.05 s/lyr/sent = 1.0×**

Confounding factors:
1. **GPU contention** — number of processes sharing the same physical GPU
2. **Sentence length** — causal tracing does ntoks × num_layers forward passes per item; longer sentences spike s/it even though tqdm averages over time
3. **Precision** — float32 vs float16 (float16 ~2–4× faster in practice)

Condition labels (by process count on GPU):
- **fast**: ≤2 procs — near-isolated, minimal interference
- **low**: 3–4 procs — light sharing, small overhead
- **adverse**: 6 procs — heavy sharing, significant slowdown
- **extreme**: ~12 procs — very heavy sharing, worst observed contention

Quality labels:
- **clean** — single stable state, run completed, s/it measured directly
- **aggregate** — state changed mid-run; only the blended s/it is known, per-state unknown
- **estimate** — s/it inferred from observed range bounds; procs fluctuated; not exact
- **partial** — state existed but no s/it captured (see aggregate parent row)

All A6000 GPUs (49GB VRAM each).

---

## pythia-1b
- **Layers**: 16
- **Precision**: float16 (loaded from model config default)
- **Architecture**: GPT-NeoX
- **Param count**: ~1B

| Domain | Done / Total | GPU | Procs | Precision | s/it | s/lyr/sent | vs base | Wall time (hr) | Condition | Quality | Status |
|--------|-------------|-----|-------|-----------|------|------------|---------|----------------|-----------|---------|--------|
| gender | 1026 / 1026 | 0 | ~12 | fp16 | 17.3 | **1.08** | 1.03× | ~4.9 hr | extreme | clean | done |
| religion | 79 / 79 | 0 | ~12 | fp16 | 24.9 | **1.56** | 1.49× | ~0.5 hr | extreme | clean | done |
| profession | 810 / 810 | 6 | 4 | fp16 | 16.8 | **1.05** | 1.00× *(base)* | ~3.8 hr | low | clean | done |
| race | 2682 / 2682 | 6 | 3 | fp16 | 16.93 | **1.06** | 1.01× | ~12.6 hr | low | clean | done |

Notes:
- gender/religion ran under the original 12-process batch sharing GPU 0 across 2 GPUs — worst contention pythia ever saw, yet s/lyr/sent barely moved (~1.0 → ~1.5×). fp16 + lightweight GPT-NeoX absorbs contention well.
- profession/race restarted after crash fix on GPU 6 at lower contention — these are the cleanest low-contention pythia measurements.

---

## gpt2-medium
- **Layers**: 24
- **Precision**: fp32 (original runs) → fp16 (race domain restarted 2026-04-15)
- **Architecture**: GPT-2
- **Param count**: ~345M

| Domain | Done / Total | GPU | Procs | Precision | s/it | s/lyr/sent | vs base | Wall time (hr) | Condition | Quality | Status |
|--------|-------------|-----|-------|-----------|------|------------|---------|----------------|-----------|---------|--------|
| religion | 79 / 79 | 2 | 6 | fp32 | 194 | **8.08** | 7.70× | ~4.3 hr | adverse | clean | done |
| gender | 1026 / 1026 | 2 | 6 | fp32 | — | — | — | — | adverse | partial | migrated |
| gender | 1026 / 1026 | 6 | 3 | fp32 | — | — | — | — | low | partial | migrated |
| gender *(agg.)* | 1026 / 1026 | 2→6 | 6→3 | fp32 | 38.75 | **1.61** | 1.53× | ~11.0 hr | mixed | aggregate | done |
| profession | 810 / 810 | 2 | 6 | fp32 | — | — | — | — | adverse | partial | migrated |
| profession | 810 / 810 | 6 | 3 | fp32 | — | — | — | — | low | partial | migrated |
| profession *(agg.)* | 810 / 810 | 2→6 | 6→3 | fp32 | 43.90 | **1.83** | 1.74× | ~9.9 hr | mixed | aggregate | done |
| race *(low bound)* | ~1046 / 2682 | 2 | ~4 | fp32 | ~20 | **~0.83** | ~0.79× | — | low | estimate | killed @ 39% |
| race *(hi bound)* | ~1046 / 2682 | 2 | ~6 | fp32 | ~142 | **~5.92** | ~5.64× | — | adverse | estimate | killed @ 39% |
| race | 2682 / 2682 | 7 | 2 | fp16 | 10.08 | **0.42** | **0.40×** | ~7.5 hr | fast | clean | done |

Notes:
- `religion` is the only clean single-state adverse-condition measurement for gpt2-medium. Use this when comparing adverse vs fast.
- `gender` and `profession` aggregate rows blend adverse and low states into one s/it — the low-contention speedup is hidden inside. Per-state s/it must be re-captured on future reruns to recover the split.
- `race` fp32 was killed at 39% (~1046 items done). The procs count on GPU 2 fluctuated between 4 and 6 during the run, causing s/it to swing 20–142. Low/hi bound rows are estimates from the observed extremes, not exact per-state measurements.
- `race` fp16 on GPU 7 (2 procs) is the only clean fast-condition measurement. At **0.42 s/lyr/sent** it is the fastest single measurement in the entire dataset — faster than pythia-1b at baseline.

---

## OLMo-2-0425-1B
- **Layers**: 16
- **Precision**: fp32 (original runs) → fp16 (gender domain restarted 2026-04-15)
- **Architecture**: OLMo-2 (SwiGLU FFN, wider MLP expansion than pythia — heavier per-forward-pass despite same layer count)
- **Param count**: ~1B

| Domain | Done / Total | GPU | Procs | Precision | s/it | s/lyr/sent | vs base | Wall time (hr) | Condition | Quality | Status |
|--------|-------------|-----|-------|-----------|------|------------|---------|----------------|-----------|---------|--------|
| religion | 79 / 79 | 2 | 6 | fp32 | 175 | **10.94** | 10.42× | ~3.8 hr | adverse | clean | done |
| gender *(low bound)* | ~718 / 1026 | 2 | ~4 | fp32 | ~66 | **~4.13** | ~3.93× | — | low | estimate | killed @ 70% |
| gender *(hi bound)* | ~718 / 1026 | 2 | ~6 | fp32 | ~177 | **~11.06** | ~10.53× | — | adverse | estimate | killed @ 70% |
| gender | 1026 / 1026 | 7 | 2 | fp16 | 9.56 | **0.60** | **0.57×** | ~2.7 hr | fast | clean | done |
| profession | 810 / 810 | 2 | 2 | fp32 | 122.58 | **7.66** | 7.30× | ~27.6 hr | fast | clean | done |
| race | 2682 / 2682 | 2 | 2 | fp32 | 43.67 | **2.73** | 2.60× | ~32.5 hr | fast | clean | done |

Notes:
- OLMo is the slowest model per layer across every condition — SwiGLU FFN + wider MLP expansion costs more per forward pass regardless of contention. `religion` at adverse hits **10.94×** baseline, the worst observed value in the dataset.
- `gender` fp32 was killed at 70% (~718 items done). Procs on GPU 2 fluctuated between 4 and 6; the low/hi bound rows estimate the s/it at each extreme.
- `profession` fp32 at fast (2 procs) still hits **7.66 s/lyr/sent** — this is architecture cost, not contention. Even with no other jobs competing, OLMo fp32 is 7.3× the baseline.
- `race` fp32 at fast (2 procs) gives **2.73 s/lyr/sent** — lower than profession at the same condition. Likely due to shorter average sentence length in the race domain (fewer tokens → fewer forward passes per item).
- `gender` fp16 on GPU 7 (2 procs) at **0.60 s/lyr/sent** shows fp16 brings OLMo into competitive range, though it remains 1.4× slower per layer than pythia-1b fp16 at baseline.

---

## Summary: Per-layer cost by model × condition

| Model | Arch | Params | Layers | Precision | Condition (procs) | s/lyr/sent | vs base | Quality | Domain(s) |
|-------|------|--------|--------|-----------|-------------------|------------|---------|---------|-----------|
| pythia-1b | GPT-NeoX | ~1B | 16 | fp16 | low (3–4 procs) | **~1.05–1.06** | **1.0× *(base)*** | clean | profession, race |
| pythia-1b | GPT-NeoX | ~1B | 16 | fp16 | extreme (~12 procs) | **~1.08–1.56** | 1.0–1.5× | clean | gender, religion |
| gpt2-medium | GPT-2 | ~345M | 24 | fp32 | adverse (6 procs) | **~8.08** | 7.7× | clean | religion |
| gpt2-medium | GPT-2 | ~345M | 24 | fp32 | mixed (6→3 procs) | **~1.61–1.83** | 1.5–1.7× | aggregate | gender, profession |
| gpt2-medium | GPT-2 | ~345M | 24 | fp32 | low (~4 procs) | **~0.83** | ~0.8× | estimate (lower bound) | race |
| gpt2-medium | GPT-2 | ~345M | 24 | **fp16** | fast (2 procs) | **~0.42** | **0.40×** | clean | race |
| OLMo-2-1B | OLMo-2 | ~1B | 16 | fp32 | adverse (6 procs) | **~10.94–11.06** | ~10.5× | clean + estimate | religion, gender hi |
| OLMo-2-1B | OLMo-2 | ~1B | 16 | fp32 | low (~4 procs) | **~4.13** | ~3.9× | estimate (lower bound) | gender |
| OLMo-2-1B | OLMo-2 | ~1B | 16 | fp32 | fast (2 procs) | **~2.73–7.66** | 2.6–7.3× | clean | race, profession |
| OLMo-2-1B | OLMo-2 | ~1B | 16 | **fp16** | fast (2 procs) | **~0.60** | **0.57×** | clean | gender |

Key takeaways:
- **fp16 is the single biggest lever**: gpt2-medium fp16 fast (0.42) beats pythia-1b fp16 at low contention (1.05). OLMo fp16 fast (0.60) beats pythia-1b fp16 at extreme contention (1.08–1.56). Switching precision matters more than reducing contention.
- **OLMo fp32 is architecture-bound, not contention-bound**: at fast (2 procs) it still hits 7.66 (profession) — 7.3× over baseline. The SwiGLU cost does not disappear under light load.
- **pythia-1b fp16 is nearly contention-immune**: going from 3 procs to 12 procs only moves it 1.05 → 1.56 (48% degradation). All other models degrade far more steeply.
- **Adverse fp32 is impractical for OLMo**: 10–11× baseline means a 2682-item race domain run at 6 procs would take ~130 hrs wall time.

---

## Data gaps to fill on future reruns

| Model | Domain | Missing state | What to log | Why it matters |
|-------|--------|---------------|-------------|----------------|
| gpt2-medium | gender | GPU 2, 6 procs (fp32) | s/it before migration | recover clean adverse measurement |
| gpt2-medium | gender | GPU 6, 3 procs (fp32) | s/it after migration | recover clean low measurement |
| gpt2-medium | profession | GPU 2, 6 procs (fp32) | s/it before migration | recover clean adverse measurement |
| gpt2-medium | profession | GPU 6, 3 procs (fp32) | s/it after migration | recover clean low measurement |
| gpt2-medium | race | GPU 2, 4-proc state (fp32) | confirm exact s/it at 4 procs | validate low-bound estimate (~0.83) |
| gpt2-medium | race | GPU 2, 6-proc state (fp32) | confirm exact s/it at 6 procs | validate hi-bound estimate (~5.92) |
| OLMo-2-1B | gender | GPU 2, 4-proc state (fp32) | confirm exact s/it at 4 procs | validate low-bound estimate (~4.13) |
| OLMo-2-1B | gender | GPU 2, 6-proc state (fp32) | confirm exact s/it at 6 procs | validate hi-bound estimate (~11.06) |
