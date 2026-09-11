# Pipeline State — Signed NIE, Cross-Model Anchoring, and What's Safe to Claim

**Date:** 2026-09-11. **Scope:** `experiments/bias_trace.py`, `plot_utils.py`, `fig.py`,
`scripts/table.py`, as of commit `86d451b` on `main`. Written to be read by whoever
writes the paper next — it says what changed, why, what you can cite as a result, and
what's still an open question rather than a settled fact.

---

## 1. Two metrics — keep them separate in your head and in the writing

Every result file now carries two independent readings of the same forward passes.
They answer different questions and should never be described with the same word.

| | **ALP** (Absolute Log-Prob Diff) | **NIE** (Normalized Indirect Effect) |
|---|---|---|
| Question it answers | *Where does bias exist* — magnitude only | *How much does restoring/corrupting a state move the model's own preference* — direction matters |
| Chain | `scores` / `high_score` / `low_score` — **absolute value** | `scores_signed` / `high_score_signed` / `low_score_signed` — **signed**, stereo-minus-anti |
| Formula | mean per-token log-prob diff, `abs()`'d | `(restored − low) / (high − low)`, all three on the signed chain |
| Where computed | `bias_trace.py`'s `causal_difference` (both chains, every case) | `plot_utils.normalized_indirect_effect(..., signed=True)` |
| Direction-aware? | No — a state that restores the anti-stereotype scores the same as one that restores the stereotype | Yes — that's the entire reason the signed chain exists |

**Why both exist, in one sentence for the paper:** ALP tells you the layer/token
*localizes* something; NIE (signed) tells you *whether restoring it pushes the model
toward or away from the stereotype it started with*. A layer can have high ALP and
near-zero signed NIE (it moves the score, but roughly as often toward the
anti-stereotype as the stereotype across cases) — that's not a contradiction, it's the
whole point of keeping the two separate.

---

## 2. What changed this session, in order

All on `main`, no separate branches. Read the commit messages for full technical
detail — this is the one-paragraph version of each.

1. **`05c6f5f` — Added the signed chain to plotting.** `collect_scores()` and
   `normalized_indirect_effect()` gained a `signed=` flag. Before this, NIE existed
   in name only — the code computed it from the abs chain, which makes it a pure
   rescaling of ALP (no new information; a state that flips the model toward the
   anti-stereotype scored identically to one that restores the stereotype).
2. **`3d00194` → reverted in `3bed2ca` — ROME Fig. 3 "severing."** Explored adding a
   necessity-test (force a state back to its corrupted value *after* restoring others,
   to ask "is this state required," not just "is it sufficient"). Built and verified
   logic-only (toy model, no GPU), but never wired to a driver or CLI — no experiment
   ever used it. Removed rather than left half-built; if this is wanted later, the
   design and the specific bug it was built to avoid (a shared-RandomState issue in
   the old, also-removed `trace_with_repatch`) are written down in `3d00194`'s message
   for reference.
3. **`634d94b` — `scripts/table.py`'s comparison table moved to signed NIE**, through
   the same shared function as everywhere else, rather than its own inline abs-chain
   computation.
4. **`7aec69b`, `89ba02e` — two bug fixes** surfaced by testing the above: a crash
   (not a graceful "n/a") on any result file without the signed chain, and a
   pre-existing, unrelated `IndexError` on single-domain figure runs.
5. **`b085ecd` — the pooling-noise-amplification fix (see §4 below).** Signed NIE
   pools raw values across cases with no per-case sign alignment (deliberate — see
   §4). That pooling can produce a near-zero *denominator* even when no individual
   case is degenerate, and dividing by a near-zero denominator manufactures large,
   fake-looking per-layer "effects." Added `signed_gap_reliable()` — reuses the
   existing `LOW_SIGNAL = 0.03` cutoff (already calibrated for the identical failure
   mode on the abs chain) as a display-time gate: a domain whose pooled signed gap
   is too small reports NIE as unavailable instead of a wild number.
6. **`e373d1c` — finished the rollout.** Five remaining figures (appendix grids,
   overlays) still computed "NIE" from the abs chain — same name, different,
   redundant-with-ALP statistic. All now use the signed chain, same guard as
   everywhere else. Consolidated four duplicated loader functions into one shared
   `_finalize_result_dict()` in the process — net **−16 lines** in `fig.py`, not more.
7. **`86d451b` — resolved a stale in-code TODO** about which model's embedding std to
   use for noise scaling (answer: always the target's — was already correct, the
   comment just still posed it as an open question).

**Net effect on `bias_trace.py` specifically: 65 lines *smaller* than before this
session** — the only lasting change to that file is deleting dead code
(`trace_with_repatch`, confirmed unused and confirmed buggy: it shared one
`RandomState` across two internal passes, so its two passes never drew the same
noise for what was supposed to be one consistent corruption).

---

## 3. What you can say in the paper now, and how

### Within-model localization (ALP + signed NIE)
Mechanically sound — verified end to end (batch convention, noise reproducibility,
patch direction, hook mechanics). Safe to report **with two caveats you should state
explicitly, not bury**:
- The corruption effect is often small or the wrong sign: on real OLMo-2-1B gender
  data, median `high − low` gap is **0.046**, and **38.6%** of cases have a
  non-positive gap (corruption didn't reduce the stereo/anti separation, or increased
  it). This is measured, not suspected — see `review.md` B2, independently
  reproduced this session from the raw `.npz` files.
- A plausible, currently untested explanation is dilution: the reported score
  averages over the *whole sentence*, so a 1–2 token fill word is diluted across a
  5–25 token context (`review.md` B3, `UNEQUAL_LENGTH_PROBLEM.md`). The blank-only
  chain (`scores_blank`, `high_score_blank`, `low_score_blank`) already exists in
  every result file specifically to test this, and has never been run as a check.
  **This is the single cheapest experiment that would tell you whether the small-gap
  problem is a metric artifact or a real, weaker finding — do this before writing the
  localization section, not after.**

### Cross-model patching (base ↔ instruct)
**Do not report cross-model NIE as a ratio, percentage, or mean/median "recovery"
number.** Verified directly this session (`verification/cross_model_scale_probe_olmo.json`,
real OLMo-2-1B base/instruct, real production code — not a toy or reimplementation):

- The patching *mechanism* is correct: restoring every state from a model's own clean
  run reproduces its own clean score to float32 precision (24/24 cases, error ~1e-6).
- Restoring every state from the *other* model's clean run does **not** land inside
  `[low, high]` most of the time (33.3% for base→instruct, 8.3% for instruct→base),
  and when it doesn't, it isn't just "outside the range a bit" — individual cases
  land at 20–28× the width of the entire clean-to-corrupted gap, in both directions
  (worse than corrupted, and more extreme than clean). The mean of the 12
  base→instruct cases is not even a stable number: dropping the single worst outlier
  flips it from −1.00 to +1.44.

**Why this isn't fixed by the pooling guard in §4:** that guard protects against a
small *denominator*. This problem shows up on cases with a perfectly healthy
denominator — it's the *numerator* (the cross-model restored value) that isn't
guaranteed to mean anything on the target model's own corrupted-to-clean scale, because
it's a value from a different model's weights, not a weaker copy of the target's own
signal. Same architecture (`validate_model_pair`'s checks) guarantees the patch is
*mechanically* valid — shapes line up — it says nothing about whether the two models'
internal representations agree on what a given activation *means*.

**What this is good evidence for, and how to phrase it:** not "bias information
transfers incompletely between training stages" (implies a measurable degree, which
the data doesn't show — it's not attenuated toward zero, it's scattered on both
sides of the entire plausible range). The defensible, and more interesting, claim is:
**forcing one training stage's subject-conditioned activations into the other
produces effects with no consistent relationship to that model's own
corrupted-to-clean scale — evidence that the two stages encode subject-conditioned
bias information in genuinely incompatible internal representations, not merely
weaker or attenuated copies of each other.** Report it with statistics that don't
assume a bounded, interpolating scale:
- Fraction of cases landing in `[low, high]` at all (33% / 8% above) — this number
  *is* the finding, not a nuisance statistic.
- The raw, unnormalized `cross_all` score plotted directly against `low`/`high`
  reference lines, rather than a normalized ratio.
- A dispersion statistic (std ≈ 8.5 on both directions above) rather than a mean or
  median.
- The direction asymmetry itself is worth a sentence: base→instruct lands in-range
  4× more often than instruct→base, mild evidence that fine-tuning adds structure the
  base model's later layers never learned to handle (harder to hand instruct's
  activations to base than the reverse), not just rotates existing structure in place.

`scripts/table.py`'s Part 2 (W1/JSD distributional distance) is on firmer ground for
a cross-model comparison than any NIE-based figure, because it compares raw pooled
distributions directly and never assumes the `[low, high]` anchoring that breaks
above.

---

## 4. Deliberate design choices — settled, but worth knowing why when someone asks

- **NIE pools raw signed values across cases before dividing once; it does not
  normalize each case by its own gap and then average ("macro" averaging).** This
  is a live tension with `BLUEPRINT.md §2`, which states the project's originally
  intended convention as macro averaging, citing Zhang & Nanda (2024) and Sen Sharma
  et al. (2024). The pooled approach was kept deliberately: a domain whose cases
  split between stereotype- and anti-stereotype-preferring should show up as a small
  net signal (a real finding — "this domain isn't uniformly stereotyped"), not be
  aligned away by normalizing each case to its own direction first. The tradeoff is
  the noise-amplification failure `signed_gap_reliable()` (§2.5) now guards against.
  If a reviewer pushes on this, the honest answer is: it's a considered choice with a
  known, guarded failure mode, not an oversight — but re-litigate it before finalizing
  if the guard ends up suppressing a large fraction of domains once real data exists.
- **`LOW_SIGNAL = 0.03`** is reused as the reliability threshold for both chains.
  `BLUEPRINT.md` itself flags that this specific number has no citable, published
  justification and should be sensitivity-checked before it appears in the paper as
  more than a display heuristic.
- **Gaussian-noise corruption only** (no symmetric-token-replacement cross-check).
  Matches the ROME/Meng et al. lineage; not revisited this session by explicit
  decision.

---

## 5. Known, already-documented issues this session did not change

Full detail in `review.md` and `UNEQUAL_LENGTH_PROBLEM.md` — not re-litigated here,
just indexed so you know where to look:

- **~11–14% of cases are causally unmeasurable by construction** (the BLANK token
  precedes every subject-span token, so no subject-side intervention can reach it in
  an autoregressive model) and are silently retained in every aggregate (`review.md`
  S3).
- **Different models see different case populations.** Cross-family case overlap for
  gender is ~83% of the union; the drop mechanism (token-length mismatch) is
  systematically biased toward dropping morphologically complex/rarer stereotype
  terms, not random (`review.md` S4, `UNEQUAL_LENGTH_PROBLEM.md`). Harmless within a
  base/instruct pair (same tokenizer); a real confound across model families.
- **bf16 quantization on every instruct-model target** (every model here loads bf16
  except the OLMo base checkpoints) may be on the same order of magnitude as the
  effect sizes being measured. Independently sanity-checked this session via a
  CPU-only simulation of bf16 rounding on realistic logit magnitudes (perturbation to
  `log_softmax` output: median 0.006–0.12, max up to 0.34, vs. a median effect gap of
  0.046) — plausible and credible, not confirmed on the live model (`review.md` B1;
  needs GPU access to close out properly).

---

## 6. One thing to check before trusting *any* number above against a fresh run

**Every currently-existing result file (local checkout, shared results root,
`results.zip`, `main.zip`) predates this session's code** — legacy format, no
`score_metric`/`scores_signed`/provenance fields, fixed `window=10` regardless of
model depth. None of §1–§4's guards or the signed chain have ever run against a real
model. The cross-model anchoring problem in §3 is the one exception — it was measured
directly from a small, real, already-executed probe
(`verification/cross_model_scale_probe_olmo.json`), not from the legacy result
corpus, so that specific finding *is* live evidence. Everything else describing "what
the current code does" is verified against the code and synthetic data, not against a
full regenerated result set. Regenerate before finalizing any headline number.

---

## 7. Quick map: concept → code

| Concept | Where |
|---|---|
| ALP / signed chain definitions | `experiments/bias_trace.py::causal_difference`, `calculate_hidden_flow` |
| `collect_scores(signed=...)` | `plot_utils.py` |
| `normalized_indirect_effect(..., signed=...)` | `plot_utils.py` |
| `signed_gap_reliable()` (the §4 guard) | `plot_utils.py`, just below `LOW_SIGNAL` |
| Shared within-model/cross-patch result-dict builder | `fig.py::_finalize_result_dict` |
| Within-model report (NIE table, trajectory) | `fig.py::save_stats_and_report`, `save_bias_trajectory` |
| Cross-model probe (source of §3's numbers) | `verification/cross_model_scale_probe.py` |
| Comparison table for the paper | `scripts/table.py` |
