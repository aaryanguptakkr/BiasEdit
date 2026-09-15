# Independent review brief — causal tracing & cross-model patching

Paste everything below the line into a fresh session. It is self-contained.

---

You are an independent reviewer with a mechanistic-interpretability background, asked to
audit the causal-tracing and activation-patching subsystem of a research codebase before an
A*-venue submission. Treat this as a peer review of *code as scientific instrument*: the
question is not "is this clean code" but **"does this code measure what the paper says it
measures, and would the results survive an expert referee?"**

Work from the code and the data. Do not take any existing document in the repo as ground
truth — several were written by people with a stake in the answer, and one is a review of an
older snapshot. Form your own view first.

## What the project does

The codebase adapts ROME-style causal tracing (Meng et al. 2022) from *factual recall* to
*social bias*. For each StereoSet intrasentence case there are two sentences that differ in
exactly one word — a stereotypical fill and an anti-stereotypical fill (e.g. "The grandfather
walks very **slow**." vs "…very **fast**."). The pipeline:

1. corrupts the subject-token embeddings with Gaussian noise,
2. restores selected internal activations,
3. scores how much of the model's stereo-vs-anti preference returns,
4. normalises that into an indirect-effect measure per (token, layer),

and does this both **within one model** and **across two models** (a base model's clean
activations patched into an instruction-tuned model's corrupted run, and vice versa), across
several model families and three bias domains.

## Where the code is

- **Authoritative checkout: `/deepfreeze/aag026/Aaryan2/BiasEdit`, branch `main`.**
  All work is merged here; review this tree.
- A second checkout exists at `/deepfreeze/aag026/Aaryan2/BiasEdit-wt-cmp` (branch
  `cross-model-patching`, fully merged into main and now behind). Ignore it unless you are
  tracing history.

**In scope** (all paths relative to `bias_tracing/`):

| file | role |
|---|---|
| `experiments/bias_trace.py` | the whole tracing pipeline: CLI, noise, hooks, patching, scoring, per-case output |
| `dsets/stereoset.py` | dataset loading; locates the token span of the differing fill word |
| `util/nethook.py` | the hooking/tracing utility everything else is built on |
| `plot_utils.py`, `fig.py` | aggregation, normalisation and every figure in the paper |
| `scripts/*.sh`, `scripts/table.py` | how runs were actually configured and tabulated |
| `data/domain/*.json`, `data/knowns.json`, `data/stereoset_subjects.json` | inputs |
| `results/**/*.npz`, `results/run_log.jsonl` | produced results, for spot-checking only |

**Out of scope:** model editing / mitigation code (`bias_tracing/mitigation/`, repo-root
editing code), and anything unrelated to tracing.

## Reference material you should check against

- ROME paper: arXiv:2202.05262 — especially §2.1–2.2, Figures 1–3, Appendix B.
- ROME code: `git clone --depth 1 https://github.com/kmeng01/rome` — note it contains
  **two** causal-tracing notebooks plus `experiments/causal_trace.py`.
- The upstream project this was forked from: `git clone --depth 1 https://github.com/zjunlp/BiasEdit`
  (its `bias_tracing/` directory is the direct ancestor of this code).
- StereoSet (Nadeem et al. 2021) for what the data actually is and how it is meant to be scored.

Divergences from these references are not automatically defects — but every divergence
should be *identified*, and you should judge whether it is an improvement, a neutral
refactor, or a silent change of meaning.

## Environment — and hard limits

```bash
cd /deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing      # required: globals.yml is read from cwd
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 conda run -n bias_trace_olmo python <script>
```
Model weights and tokenizers are cached locally (OLMo-2-0425-1B and -Instruct, pythia-1b,
Llama-3.2-1B, gemma-3-1b-pt/it, Qwen2.5, gpt2). Tokenizer-only and meta-device checks are
cheap — use them liberally to turn arguments into measurements. A 1B model will run on CPU
if you need a handful of real forward passes.

**Do not:**
- launch GPU tracing runs, training, or any long job (GPUs are shared; 0 and 4–7 belong to
  other people). If a finding needs a GPU run to confirm, say so and stop.
- modify, delete or overwrite any file in `results/`, or any tracked source file.
- commit, push, stash, or change git state in any way.
- write anywhere except a scratch directory of your own.

Read-only analysis plus scratch scripts is the whole job.

## What to examine

Answer these by reading code and running cheap checks — not from general knowledge of how
these libraries usually behave.

**A. The intervention.** How is corruption implemented, and on what exactly? How are clean
activations obtained and injected, and is that correct when the donor and recipient are
*different models*? Trace the batch convention (which rows are clean, which corrupted) all
the way through to the score. Is the randomness reproducible, and is it the *same* randomness
everywhere it needs to be? What is hooked for each intervention variant, and does the hooked
module correspond to the quantity named in the code, the plots, and the paper?

**B. The metric.** Which token positions enter the score, and which are excluded — trace the
label tensor from construction to use, accounting for the causal-LM shift. Is the quantity a
sum, a mean, a probability, a log-probability? Is the aggregation over noise samples, over
cases, and over tokens done in a defensible order? How is the normalised effect defined, is
it defined identically everywhere it appears, and what happens in degenerate cases? Does the
metric as implemented match the metric as it would be written in a paper?

**C. Data handling and selection.** How is the differing fill word located in token space,
and how is the subject located? Enumerate **every** path by which a case can be skipped or
dropped, and **quantify each one** — how many cases, in which domains, for which tokenizers,
and is the loss random or correlated with a property of the data? Does the surviving sample
support the comparisons being made across models and across domains?

**D. Experimental design.** For each experiment the pipeline runs, state precisely what
intervention it performs and what claim that intervention can support. Are there experiments
named in the code or figures that are never executed? Are there parameters inherited from the
reference implementation whose value only makes sense for the original model's architecture?

**E. Analysis and figures.** Does each plotted series come from the data its label claims?
How are per-case values combined, and does that weighting bias comparisons between models
whose tokenizers segment words differently? Is any uncertainty reported? Would a referee be
able to tell from the figures what was actually computed?

**F. Reproducibility.** Can a result file be traced back to the exact code, model, direction,
and parameters that produced it? What happens if the pipeline is re-run over existing results?
Are the declared dependencies the ones actually used?

## Standard of evidence

- Anchor every finding to `file:line`, or to output from a check you ran (include the command).
- Separate **verified** (you observed it) from **suspected** (it looks wrong but you could not
  confirm). Never present the second as the first.
- Quantify wherever a number is possible: "N of M cases", not "many cases".
- If you claim behaviour differs from a reference implementation, quote both.
- Actively try to falsify your own findings before reporting them. A confident wrong finding
  costs more than a missed one here.

## Output

A single report, findings ordered by scientific severity:

1. **Verdict** — one paragraph: would these results survive expert review as they stand?
2. **Findings**, each with: location · what is wrong · **why it matters scientifically** (what
   claim it threatens) · evidence · suggested fix · cost of that fix · your confidence.
   Rank: *invalidates a claim* > *biases a result* > *limits interpretation* > *hygiene*.
3. **What is correct** — briefly note the parts you checked and found sound, so effort is not
   spent re-verifying them.
4. **Could not verify** — what you were unable to check, and what access would be needed.
5. **Prioritised next steps** — what to fix first, and what can be disclosed rather than fixed.

Only after your report is written: read `bias_tracing/REVIEW_CHECKLIST.md`,
`bias_tracing/points_to_discuss.md`, and any `*_REPORT.md` / `SCORING_VERIFICATION.md` in the
repo, and append a short section listing (a) findings of yours already known there, (b)
claims there you believe are wrong, and (c) anything they cover that you missed. Do not read
these before forming your own conclusions.
