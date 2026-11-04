# The effect gap in RQ3 is signed

*2026-11-03*

The effect gap in the paper is the mean signed LPD of the clean run minus that of the corrupted run, with no absolute values:

$$\textit{Effect Gap} = \frac{1}{N}\sum_{n=1}^{N}\bigl(\textit{LPD}_n^\text{clean} - \textit{LPD}_n^\text{corrupted}\bigr), \qquad \textit{LPD} = \log P_\theta(x_\text{stereo}) - \log P_\theta(x_\text{anti}).$$

The same signed difference is the denominator of NIE.

## Where the code computes it

| Step | File and lines | Code |
| --- | --- | --- |
| 1. Signed LPD of the clean run, per sample | `experiments/bias_trace.py` 309 | `base_score_signed = causal_difference(...)` |
| 2. Signed LPD of the corrupted run, per sample | `experiments/bias_trace.py` 343 | `low_score_signed = causal_difference(...)` |
| 3. Both saved in every result file | `experiments/bias_trace.py` 422–423 | `result["high_score_signed"]`, `result["low_score_signed"]` |
| 4. Effect gap for the paper's tables | `scripts/tables.py` 209–210 | `s = run['sgn']` · `gap = bmean(w, s['high']) - bmean(w, s['low'])` |
| 5. NIE, divided by the same gap | `scripts/tables.py` 135–137 | `(mean restored − low) / (high − low)`, all on `run['sgn']` |

`causal_difference` (`experiments/bias_trace.py` 589–606) returns the mean token log-probability of the stereotyped context minus that of the anti-stereotyped one. Steps 1–3 cite commit `06968e1`. That commit is the same, for these lines, as the copy that produced the results (`xuxin_transfer/aag026_bias_tracing/repo`).

## Check on real data

For OLMo 2 1B Pre and gender, recomputed from the 687 result files:

| | Clean mean | Corrupted mean | Gap |
| --- | --- | --- | --- |
| Signed LPD (used in the paper) | 0.2409 | 0.1398 | **0.1011** |
| abs. LPD (not used) | 0.4025 | 0.3366 | 0.0659 |

The paper reports 0.101 for this case (`tab:rq3_loc`), which is the signed gap. An independent check with its own code reproduced the signed gaps for OLMo 2, Qwen2.5 and Pythia exactly.

## Notes

- **Figures:** the figures in the paper use the signed gap. In commit `06968e1`, the A5 effect-gap figures still plotted the abs. LPD gap. The figure code was updated after that commit, and the figures in the paper were made with the updated code.
- **Low-signal threshold:** a bias type is flagged when the absolute value of its signed effect gap is below 0.03.
- **To do:** `scripts/tables.py`, which produces every RQ3 table, is not committed yet.
