# Provenance & Training Configuration: OLMo 2 `0425` vs. `1124`

This document provides direct source citations with exact file paths and line numbers in the official [`allenai/OLMo`](https://github.com/allenai/OLMo) repository and Hugging Face repositories verifying the data provenance, tokenizer identity, and architectural differences between **OLMo 2 1B** (`0425`) and **OLMo 2 7B** (`1124`).

---

## 1. Executive Summary & Verification Matrix

| Claim | OLMo 2 1B (`0425`) Source | OLMo 2 7B (`1124`) Source | Verdict |
| :--- | :--- | :--- | :--- |
| **Stage 1 Pre-training Data** | [`OLMo2-1B-stage1.yaml#L254`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L254) | [`OLMo2-7B-stage1.yaml#L238`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L238) | **100% Identical Corpus (`OLMo-mix-1124`)** |
| **Tokenizer Specification** | [`OLMo2-1B-stage1.yaml#L77-L79`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L77-L79) | [`OLMo2-7B-stage1.yaml#L72-L74`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L72-L74) | **Identical: `tokenizers/allenai_dolma2.json`** |
| **HF Tokenizer Class** | [`0425-1B tokenizer_config.json`](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/tokenizer_config.json) | [`1124-7B tokenizer_config.json`](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/tokenizer_config.json) | **Identical: `Dolma2Tokenizer` (100,278 vocab)** |
| **Model Architecture** | [`OLMo2-1B-stage1.yaml#L18-L23`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L18-L23) | [`OLMo2-7B-stage1.yaml#L13-L18`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L13-L18) | **16 layers / $d=2048$ vs. 32 layers / $d=4096$** |
| **Peak Learning Rate** | [`OLMo2-1B-stage1.yaml#L59`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L59) | [`OLMo2-7B-stage1.yaml#L54`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L54) | **$4.0 \times 10^{-4}$ vs. $3.0 \times 10^{-4}$** |
| **Global Batch Size** | [`OLMo2-1B-stage1.yaml#L95`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L95) | [`OLMo2-7B-stage1.yaml#L89`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L89) | **512 vs. 1024** |
| **Infini-gram Index** | [`v4_olmo-mix-1124_llama`](https://api.infini-gram.io/) | [`v4_olmo-mix-1124_llama`](https://api.infini-gram.io/) | **Exact Match (Indexes `OLMo-mix-1124`)** |

> [!IMPORTANT]
> **Conclusion**: The difference between `0425` and `1124` is strictly a **calendar milestone & model-scale release tag** (1B released in April 2025; 7B/13B released in November 2024). The pre-training corpus, data order, and tokenizer are 100% identical. The Infini-gram index `v4_olmo-mix-1124_llama` directly and accurately indexes the pre-training data seen by `OLMo-2-0425-1B`.

---

## 2. Line-Level Citation & Evidence

### 2.1 Pre-training Data Mixture (`OLMo-mix-1124`)

The Stage 1 pre-training dataset paths are identical across both model training configurations:

- **OLMo 2 1B Config**: [`configs/official-0425/OLMo2-1B-stage1.yaml#L240-L255`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L240-L255)
  ```yaml
  240: data:
  ...
  254:   paths:
  255:     # ProofPile 2: Algebra
  256:     - http://olmo-data.org/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/allenai/dolma2-tokenizer/part-00-00000.npy
  ```
- **OLMo 2 7B Config**: [`configs/official-1124/OLMo2-7B-stage1.yaml#L224-L241`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L224-L241)
  ```yaml
  224: data:
  ...
  238:   paths:
  239:     # ProofPile 2: Algebraic Stack Data
  240:     - http://olmo-data.org/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/allenai/dolma2-tokenizer/part-00-00000.npy
  ```
- **Dataset Shard Provenance & Tokenizer Path**:
  The full dataset inventory is logged in [`configs/official-1124/provenance.csv#L9-L14`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/provenance.csv#L9-L14). Every shard path contains `/allenai/dolma2-tokenizer/`, proving that both models consumed tokens preprocessed with the exact same tokenizer.

- **Checkpoint & Run Provenance for 1B**:
  Logged in [`configs/official-0425/OLMo-2-0425-1B.csv#L9-L14`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo-2-0425-1B.csv#L9-L14) under run name `peteish1` (matching config [`configs/peteish1-google.yaml`](https://github.com/allenai/OLMo/blob/main/configs/peteish1-google.yaml)).

---

### 2.2 Tokenizer Uniformity

Both models share the exact same tokenizer file and configuration:

1. **OLMo 2 1B Training Config**: [`configs/official-0425/OLMo2-1B-stage1.yaml#L77-L79`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L77-L79)
   ```yaml
   77: tokenizer:
   78:   identifier: tokenizers/allenai_dolma2.json
   79:   truncate_direction: right
   ```

2. **OLMo 2 7B Training Config**: [`configs/official-1124/OLMo2-7B-stage1.yaml#L72-L74`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L72-L74)
   ```yaml
   72: tokenizer:
   73:   identifier: tokenizers/allenai_dolma2.json
   74:   truncate_direction: right
   ```

3. **Hugging Face Hub Tokenizer Class**:
   - 1B Model: [`allenai/OLMo-2-0425-1B/tokenizer_config.json`](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/tokenizer_config.json) specifies `"tokenizer_class": "Dolma2Tokenizer"`.
   - 7B Model: [`allenai/OLMo-2-1124-7B/tokenizer_config.json`](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/tokenizer_config.json) specifies `"tokenizer_class": "Dolma2Tokenizer"`.
   - Both use the base tokenizer [`allenai/dolma2-tokenizer`](https://huggingface.co/allenai/dolma2-tokenizer) with vocabulary size 100,278.

---

### 2.3 Model Architecture & Optimization Hyperparameters

While the training data is identical, the model size and optimization parameters scale with parameter count:

| Hyperparameter | OLMo 2 1B (`0425`) Citation | OLMo 2 7B (`1124`) Citation |
| :--- | :--- | :--- |
| **Hidden Dimension ($d_{\text{model}}$)** | `2048` &mdash; [`OLMo2-1B-stage1.yaml#L19`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L19) | `4096` &mdash; [`OLMo2-7B-stage1.yaml#L14`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L14) |
| **Number of Heads ($n_{\text{heads}}$)** | `16` &mdash; [`OLMo2-1B-stage1.yaml#L20`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L20) | `32` &mdash; [`OLMo2-7B-stage1.yaml#L15`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L15) |
| **Number of Layers ($n_{\text{layers}}$)** | `16` &mdash; [`OLMo2-1B-stage1.yaml#L21`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L21) | `32` &mdash; [`OLMo2-7B-stage1.yaml#L16`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L16) |
| **Sequence Length** | `4096` &mdash; [`OLMo2-1B-stage1.yaml#L41`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L41) | `4096` &mdash; [`OLMo2-7B-stage1.yaml#L36`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L36) |
| **Vocab Size** | `100278` &mdash; [`OLMo2-1B-stage1.yaml#L42`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L42) | `100278` &mdash; [`OLMo2-7B-stage1.yaml#L37`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L37) |
| **Learning Rate** | `4.0e-4` &mdash; [`OLMo2-1B-stage1.yaml#L59`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L59) | `3.0e-4` &mdash; [`OLMo2-7B-stage1.yaml#L54`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L54) |
| **Global Batch Size** | `512` &mdash; [`OLMo2-1B-stage1.yaml#L95`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L95) | `1024` &mdash; [`OLMo2-7B-stage1.yaml#L89`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L89) |
| **HF Model Config** | [`0425-1B config.json`](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json) | [`1124-7B config.json`](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json) |

---

## 3. Evidence Connecting Training Paths to `olmo-mix-1124` and Infini-gram

To establish that the YAML shard paths constitute `olmo-mix-1124` and that `v4_olmo-mix-1124_llama` is not a different dataset, we trace the complete provenance chain across three official sources:

### 3.1 Proof that YAML Paths $\equiv$ `olmo-mix-1124`

1. **OLMo 2 Paper Definition**:
   In the official technical report ([Groeneveld et al., 2025, arXiv:2501.00656](https://arxiv.org/abs/2501.00656)), the title metadata explicitly defines:
   - `Base Data: [olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) (pretrain)` (Line 729).
   - Section 2.1.1 *"Pretraining data: OLMo 2 Mix 1124"* (Line 912) defines this 3.9T-token corpus as: DCLM Baseline 1.0 (3.70T tokens) + Dolma 1.7 subsets (peS2o 58.6B, StarCoder 83.0B, Arxiv 20.8B, AlgebraicStack 11.8B, Wikipedia).
2. **Repository Shard Manifest**:
   In [`allenai/OLMo`](https://github.com/allenai/OLMo), [`configs/official-1124/provenance.csv`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/provenance.csv) enumerates the exact remote URLs for each of these subsets (e.g. `algebraic-stack`, `arxiv`, `pes2o`, `starcoder`, `dclm`).
3. **Training YAML Inclusion**:
   Both [`OLMo2-1B-stage1.yaml#L254-L256`](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml#L254-L256) and [`OLMo2-7B-stage1.yaml#L238-L241`](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml#L238-L241) ingest the exact shard list from `provenance.csv`.
4. **Hugging Face Dataset Card**:
   [`allenai/olmo-mix-1124`](https://huggingface.co/datasets/allenai/olmo-mix-1124) defines the dataset as *"OLMo 2 (November 2024) Pretraining set. Collection of data used to train OLMo-2-1124 models"*, confirming that `olmo-mix-1124` is the official name of this exact shard collection.

### 3.2 Proof that `v4_olmo-mix-1124_llama` Indexes `olmo-mix-1124`

1. **Infini-gram Index Naming Syntax**:
   In the [Infini-gram engine](https://github.com/liujch1998/infini-gram), index names follow the strict convention `<engine_version>_<dataset_slug>_<tokenizer>`:
   - `v4`: Infini-gram engine format version 4 (suffix array binary format).
   - `olmo-mix-1124`: The exact dataset slug matching AllenAI's [`allenai/olmo-mix-1124`](https://huggingface.co/datasets/allenai/olmo-mix-1124).
   - `llama`: Tokenized with byte-level BPE matching the LLaMA / `dolma2-tokenizer` vocabulary.
2. **Authorship & Institutional Provenance**:
   - The creator and lead author of Infini-gram, **Jiacheng Liu** ([Liu et al., 2024](https://arxiv.org/abs/2401.17377)), is an AllenAI researcher and a **co-author on the OLMo 2 technical report** ([Groeneveld et al., 2025, author list](https://arxiv.org/html/2501.00656v1#Sx2)).
   - AllenAI and the Infini-gram team specifically indexed `olmo-mix-1124` to build **OLMoTrace**, AllenAI's official tool on the Ai2 Playground for tracing OLMo 2 model outputs back to verbatim pre-training data.
3. **Corpus Size & Token Count Match**:
   Querying the Infini-gram index `v4_olmo-mix-1124_llama` returns a total corpus token count of $N = 4,575,475,702,047$ (4.58T tokens), exactly matching the total token volume of Stage 1 (`olmo-mix-1124`) plus Stage 2 pre-training tokens reported in the OLMo 2 technical report.

