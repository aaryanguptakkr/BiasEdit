# Cross-Model Patching Conventions: Qwen2.5-1.5B / Llama-3.2-1B / Gemma-3-1B

Decisions for extending the OLMo/Pythia cross-patching pipeline to Qwen 2.5, Llama 3.2,
and Gemma 3, grounded in a verified survey of established repos and papers
(deep-research run 2026-07-19; 25 claims adversarially verified, 21 confirmed 3-0).
Governing principle: **existing code paths (OLMo/Pythia/GPT-2) stay byte-identical;
new models get additive branches; every convention follows established practice.**

## Decisions

### 1. BOS / prefix tokens — follow each tokenizer's native convention

| Model | Convention | Mechanism |
|---|---|---|
| Qwen2.5-1.5B (+Instruct) | **no token prepended** | tokenizer adds nothing; code prepend-gate intentionally not extended |
| Llama-3.2-1B (+Instruct) | BOS at position 0 | tokenizer auto-adds `<\|begin_of_text\|>` |
| gemma-3-1b-pt / -it | BOS at position 0 (required) | tokenizer auto-adds `<bos>` |
| (existing) OLMo | manual prepend | unchanged |
| (existing) Pythia | nothing (same class as Qwen) | unchanged |

Evidence:
- TransformerLens `loading_from_pretrained.py` hardcodes `default_prepend_bos=False` for
  all Qwen architectures (deliberate exemption from its global True default); Llama-3/Gemma
  fall through to True. (verified 3-0 against live main, 2026-07-19)
- Qwen team (HF discussion Qwen2-7B-Instruct #15, org member jklj077): "There is no
  bos_token for Qwen models. It is not necessary to prepend a control token to every input
  sequence." `tokenizer_config.json`: `bos_token: null`, `add_bos_token: false`.
- lm-evaluation-harness hardcodes `add_bos_token=True` for Gemma (BOS is required for Gemma).
- ROME/MEMIT themselves implement no BOS logic — they inherit the HF tokenizer default,
  which is exactly what we do for the new families.

Safety of the no-BOS path for Qwen: `causal_difference` shifts targets
(`pred[:, :-1]` vs `targ[:, 1:]`), so the position-0 label is never scored regardless of
whether position 0 is a special token (OLMo-style) or content (Pythia/Qwen-style).

Paper caveat (cite Sun et al., "Massive Activations in LLMs", COLM 2024, App. A.3):
attention-sink "massive activations" concentrate on the first token in a
model-family-dependent way and can shift onto BOS; position 0 is excluded from corruption
and scoring in all conditions.

### 2. Locating the StereoSet BLANK — empirical length-difference, no placeholder token

For the new models the fill-word token span is computed **directly on the real sentence**:

```
start = len(encode(context_before_word))          # BOS included on both sides -> cancels
end   = len(encode(context_before_word + word))
```

- This is EasyEdit's method (`easyeditor/models/rome/repr_tools.py`,
  `get_words_idxs_in_templates`, lines 41-109), which replaced ROME's space-gluing
  heuristic; it is tokenizer-agnostic by construction and BOS cancels in the subtraction.
  (verified 3-0, twice: local clone + live main)
- **Rejected:** reserved-special-token placeholders (`<|reserved_special_token_0|>`,
  `<|fim_pad|>`). The survey found **no repo** that does this — it would be an invented
  convention. ROME/MEMIT use no placeholder at all; upstream zjunlp/BiasEdit uses
  unk-masking only for masked LMs and direct word substitution for causal LMs.
- The unk-mask path (`stereoset.py` -> `make_inputs` unk search) remains untouched for
  OLMo/Pythia/GPT-2. Note Qwen2.5 and Llama-3.2 have `unk_token = None`, so the old path
  cannot run on them even in principle.
- Each computed span is validated by decoding: `decode(ids[start:end]).strip() == word`;
  mismatching samples are logged with their ID and skipped (per-domain coverage reported).

### 3. Noise calibration — per-model embedding std (unchanged)

`s3` = 3x the std of the **target** model's own subject embeddings
(`collect_embedding_std`), applied only to subject-span rows `x[1:, b:e] += noise`.
This is the ROME and MEMIT convention verbatim (verified 3-0 in both repos); verifiers
endorsed per-checkpoint calibration as the correct extension for base<->instruct
cross-patching. The two patching directions therefore use different noise magnitudes —
by design, since corruption happens in the target's embedding space.

### 4. Cross-model (base<->instruct) patching — precedent

**CMAP** (Cross-Model Activation Patching): Prakash, Rott Shaham, Haklay, Belinkov, Bau —
"Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking", ICLR 2024
(arXiv:2402.14811), code `github.com/Nix07/finetuning`. Patches activations between base
Llama-7B and fine-tuned variants — the citable precedent that same-architecture
base<->tuned activations are compatible. No established convention exists for dtype/hook
alignment in cross-patching (our pairs are same-architecture, so hooks/dims align
trivially; both sides loaded in bf16).

### 5. Model-specific facts

| | Qwen2.5-1.5B | Llama-3.2-1B | gemma-3-1b |
|---|---|---|---|
| layers | 28 | 16 | 26 |
| dtype | bf16 | bf16 | bf16 |
| unk_token | None | None | `<unk>` |
| pad_token | `<\|endoftext\|>` | None -> set pad=eos | `<pad>` |
| BOS behavior | none (by design) | auto-added | auto-added (required) |
| module paths | `model.layers.N{.self_attn,.mlp}` (all three; existing `layername()` llama branch covers them) | | |
| HF gating | open | **gated (pending approval)** | gated (granted) |

`window=10` kept fixed for comparability with existing OLMo/Pythia results (ROME
precedent); depth fraction differs across models (10/16 vs 10/26 vs 10/28) — disclose.
Per-case `.npz` caching means a scaled-window ablation can be added later incrementally.

### 6. Known risks / open items

- **Gemma + `find_token_range` (subject location):** ROME's char-join over per-token
  decodes can misalign on SentencePiece tokenizers whose single-token decode drops leading
  whitespace; multi-word subjects are the risk case. No established repo has run ROME-style
  tracing on Gemma (EasyEdit ships no Gemma ROME config — we are first). The sanity check
  must explicitly validate subject spans on Gemma, including multi-word subjects; fallback
  fix if it fails: HF `return_offsets_mapping` based location.
- No verified precedent exists for masked/placeholder-token scoring of StereoSet blanks on
  causal LMs — our pipeline's use of the masked sentence purely for *locating* the blank
  (never scored, never fed to the model) should be stated in the paper to preempt the
  question; for new models even that is replaced by the length-difference method.
- Full deep-research report with all sources:
  session task output `wnmoajod5` (findings, refuted claims, source list).

## Sources (primary, verified)

- github.com/TransformerLensOrg/TransformerLens — `transformer_lens/loading_from_pretrained.py`
- huggingface.co/Qwen/Qwen2-7B-Instruct/discussions/15; Qwen2.5-1.5B `tokenizer_config.json`
- github.com/kmeng01/rome, github.com/kmeng01/memit — `experiments/causal_trace.py`
- github.com/zjunlp/EasyEdit — `easyeditor/models/rome/repr_tools.py`; `hparams/ROME/`
  (llama3-8b, llama3.2-3b, qwen2.5-7b exist; no Gemma)
- github.com/McGill-NLP/bias-bench — `bias_bench/benchmark/stereoset/{stereoset,dataloader}.py`
- github.com/moinnadeem/StereoSet — `code/eval_generative_models.py`
- arXiv:2402.14811 (CMAP, ICLR 2024) + github.com/Nix07/finetuning
- arXiv:2402.17762 (Massive Activations, COLM 2024)
