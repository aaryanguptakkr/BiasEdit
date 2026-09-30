# Infini-gram Co-occurrence Analysis of StereoSet Bias in OLMo 2 1B Training Data

## 1. Executive Summary & Objective

This module analyzes the pre-training foundations of social and demographic bias in **OLMo 2 1B** ([`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B)) by measuring the co-occurrence frequency, Pointwise Mutual Information (PMI), and statistical association of subject entities and attribute words from the **StereoSet** benchmark ([Nadeem et al., 2021](https://arxiv.org/abs/2004.09456)) within OLMo 2's pre-training corpus.

Pre-training co-occurrence metrics are evaluated directly against the model's actual stereotype preference scores and log-likelihood margins to determine the degree to which downstream model stereotypes reflect pre-training statistics.

> [!IMPORTANT]
> **Co-occurrence Definition**:
> Co-occurrence is computed between the **subject/target concept** and the **attribute word** (e.g., `"student"` and `"lazy"` vs. `"student"` and `"hardworking"`), **NOT** on raw sentences or prompt templates. Template sentences are sparse in pre-training corpora, whereas subject–attribute co-occurrence directly captures the semantic associations learned by the language model.

This investigation utilizes the **[Infini-gram](https://infini-gram.io/)** engine ([Liu et al., 2024](https://arxiv.org/abs/2401.17377)), querying the pre-training index `v4_olmo-mix-1124_llama` (4.58T tokens).

---

## 2. Training Data Provenance & Corpus Index

A comprehensive audit of the dataset compositions, tokenizer specifications, and remote shard paths between OLMo 2 models is documented in:
**[`CONSIDERATIONS_OLMO2_0425_VS_1124.md`](file:///deepfreeze/oyahia/BiasEdit/bias_co_ocurrence/CONSIDERATIONS_OLMO2_0425_VS_1124.md)**.

### Key Equivalence Points:
1. **Identical Pre-training Corpus**: Both `OLMo-2-0425-1B` and `OLMo-2-1124-7B` specify identical paths across all 1,122 Stage 1 parts (`OLMo-mix-1124`) and 576 Stage 2 parts (`Dolmino-mix-1124`).
2. **Milestone Nomenclature**: `0425` (April 2025) and `1124` (November 2024) denote release milestone dates and hyperparameter scale profiles, not dataset divergence.
3. **Tokenizer Uniformity**: Both use the identical [`allenai/dolma2-tokenizer`](https://huggingface.co/allenai/dolma2-tokenizer) (100,278 vocab).
4. **Infini-gram Index**: The index **`v4_olmo-mix-1124_llama`** indexes the raw text of `OLMo-mix-1124` (4.58T tokens), providing an exact representation of the pre-training corpus seen by OLMo 2 1B.

---

## 3. Dataset Ingestion & Pair Formulation

Subject and attribute pairs are ingested from `bias_tracing/data/domain/` across all four canonical StereoSet domains:
- **`gender`**: 1,026 pairs (e.g., `"female"` $\rightarrow$ `"hysterical"` vs. `"strong"`)
- **`profession`**: 810 pairs (e.g., `"civil servant"` $\rightarrow$ `"bureaucrat"` vs. `"beekeeper"`)
- **`race`**: 2,682 pairs (e.g., `"Ethiopian"` $\rightarrow$ `"skinny"` vs. `"fat"`)
- **`religion`**: 79 pairs (e.g., `"Muslim"` $\rightarrow$ `"terrorist"` vs. `"pacifist"`)
- **Total**: **4,597 pairs**

### Query Construction
For each subject $S$, stereotype attribute $W_{\text{stereo}}$, and anti-stereotype attribute $W_{\text{anti}}$:
1. **Unigram Baselines**: $C(S)$, $C(W_{\text{stereo}})$, $C(W_{\text{anti}})$
2. **Co-occurrence Within Window** ($\text{max\_diff\_tokens} = 50$):
   - Stereotype co-occurrence: $C(S \land W_{\text{stereo}})$
   - Anti-stereotype co-occurrence: $C(S \land W_{\text{anti}})$

---

## 4. Query Engine & Cache Architecture

### 4.1 Full Precision Queries
All conjunction queries are queried at full precision without lossy downsampling:
```python
payload = {
    "index": "v4_olmo-mix-1124_llama",
    "query_type": "count",
    "query": "she AND hysterical",
    "max_diff_tokens": 50,
    "max_clause_freq": 500000  # Maximum precision ceiling
}
```

### 4.2 SQLite Cache Deduplication & Resumability
- **Query Deduplication**: Many subjects (`"female"`, `"he"`, `"nurse"`) and attributes (`"strong"`, `"lazy"`) repeat across the 4,597 pairs. Caching reduces API queries from ~23,000 down to ~7,850 unique requests (~66% reduction).
- **Persistent Cache**: Queries and counts persist in [`outputs/cooccurrence/cache.db`](file:///deepfreeze/oyahia/BiasEdit/outputs/cooccurrence/cache.db).
- **Streaming JSONL Append**: Completed pairs stream to [`outputs/cooccurrence/olmo2_stereoset_cooccurrences.jsonl`](file:///deepfreeze/oyahia/BiasEdit/outputs/cooccurrence/olmo2_stereoset_cooccurrences.jsonl) with thread-safe file locks, allowing runs to be interrupted and resumed without loss.

---

## 5. Statistical Association Metrics

For total pre-training tokens $N = 4,575,475,702,047$:

### 5.1 Symmetric Dirichlet Prior Smoothing ($\text{Dir}(\alpha=1)$ / Add-1 Prior)
Raw PMI undefined on zero co-occurrences ($\log_2(0)$) forces nearly half of all pairs to be discarded. To preserve the full dataset and regularize low-frequency noise, a symmetric Dirichlet prior is applied:

$$\text{PMI}_{\text{smoothed}}(S, W) = \log_2 \frac{(C(S \land W) + 1) \cdot N}{(C(S) + 1)(C(W) + 1)}$$

$$\Delta\text{PMI}_{\text{smoothed}} = \text{PMI}_{\text{smoothed}}(S, W_{\text{stereo}}) - \text{PMI}_{\text{smoothed}}(S, W_{\text{anti}})$$

### 5.2 Raw (Unsmoothed) Baseline
For direct baseline comparison, raw Maximum Likelihood Estimation (MLE) PMI is also computed:

$$\text{PMI}_{\text{raw}}(S, W) = \log_2 \frac{C(S \land W) \cdot N}{C(S) \cdot C(W)} \quad (\text{null if } C(S \land W) = 0)$$

$$\Delta\text{PMI}_{\text{raw}} = \text{PMI}_{\text{raw}}(S, W_{\text{stereo}}) - \text{PMI}_{\text{raw}}(S, W_{\text{anti}})$$

Both metrics are recorded on every JSONL record.

---

## 6. Primary Evaluation Metrics

Pre-training $\Delta\text{PMI}$ is evaluated against model predictions using two primary metrics:

1. **Pearson Correlation Coefficient ($r$)**:
   Evaluates linear alignment between continuous training $\Delta\text{PMI}$ and continuous model preference margin $\Delta\text{LL} = \text{stereo\_ll} - \text{anti\_ll}$ via [`scipy.stats.pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).

2. **Area Under the ROC Curve (AUROC)**:
   Evaluates the discriminative ability of training $\Delta\text{PMI}$ to predict whether the model prefers the stereotype ($\text{stereo\_ll} > \text{anti\_ll}$) via [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
   - **Probabilistic Meaning**:
     $$\text{AUROC} = P\big(\Delta\text{PMI}_{\text{model picked stereo}} > \Delta\text{PMI}_{\text{model picked anti}}\big)$$
   - **Base-rate Invariant**: Completely immune to skew in the model's base rate preference (unlike raw concordance or accuracy).
   - $0.50$ indicates random chance; $1.00$ indicates perfect ranking alignment.

---

## 7. Software Architecture

Implementation organized under `bias_co_ocurrence/`:

```
bias_co_ocurrence/
├── __init__.py
├── client.py        # Infini-gram API client with SQLite cache, exponential backoff (403/429), and retry guards
├── stereoset.py     # Pair extraction, unigram/joint counting, raw & Dirichlet PMI, streaming JSONL output
├── correlate.py     # Coordinator-Worker correlation engine (scipy.stats, sklearn.metrics, pandas)
├── __main__.py      # Top-level CLI entry point
scripts/
└── run_stereoset_tmux.sh  # Automated detached tmux execution runner
outputs/
└── cooccurrence/
    ├── olmo2_stereoset_cooccurrences.jsonl  # Full 4,597-row dataset with raw & smoothed PMI
    ├── correlation_summary.json            # Final aggregate JSON report (Pearson r & AUROC)
    ├── cache.db                            # SQLite query cache (6,686+ queries)
    └── run.log                             # Execution stdout/stderr log
```

---

## 8. Results Summary

### 8.1 Full Dataset Coverage (4,597 Pairs)

| Domain | Total Pairs | Non-Zero Pairs (Raw) | Zero-Count Preserved (Smoothed) |
|---|---|---|---|
| **Gender** | 1,026 | 721 (70.3%) | **1,026 (100.0%)** |
| **Profession** | 810 | 490 (60.5%) | **810 (100.0%)** |
| **Race** | 2,682 | 1,075 (40.1%) | **2,682 (100.0%)** |
| **Religion** | 79 | 51 (64.6%) | **79 (100.0%)** |
| **TOTAL** | **4,597** | **2,337 (50.8%)** | **4,597 (100.0%)** |

> **Key Finding**: Without Dirichlet smoothing, **2,260 pairs (49.2%)** are dropped due to 0 co-occurrences. Dirichlet smoothing preserves 100% of the benchmark.

### 8.2 Instance-Level Correlation & AUROC (1,774 Evaluated Instances)

From [`outputs/cooccurrence/correlation_summary.json`](file:///deepfreeze/oyahia/BiasEdit/outputs/cooccurrence/correlation_summary.json):

| Domain | Matched Instances ($N$) | Pearson $r$ (Smoothed) | Pearson $r$ (Raw) | AUROC (Smoothed) | AUROC (Raw) |
|---|---|---|---|---|---|
| **OVERALL** | **1,774** | **$+0.0782$** ($p < 0.001$) | **$+0.0951$** ($p < 0.001$) | **0.5369** | **0.5371** |
| **Gender** | 255 | **$+0.1993$** | **$+0.1333$** | **0.5779** | **0.5131** |
| **Profession** | 810 | **$+0.1303$** | **$+0.1460$** | **0.5714** | **0.5729** |
| **Race** | 630 | $-0.0164$ | $+0.0108$ | 0.4808 | 0.5148 |
| **Religion** | 79 | $-0.0836$ | $-0.1520$ | 0.4603 | 0.4258 |

- **Gender** exhibits the strongest positive correlation with training co-occurrence ($r = +0.1993$, $\text{AUROC} = 0.5779$).
- **Profession** demonstrates consistent positive alignment across continuous margin ($r = +0.1303$) and binary classification ($\text{AUROC} = 0.5714$).
- **Overall** correlation across all 1,774 instances is positive and statistically significant ($r = +0.0782, p < 0.001$; $\text{AUROC} = 0.5369$).

---

## 9. References & Related Documents

- **Hardware & Corpus Considerations**: [`CONSIDERATIONS_OLMO2_0425_VS_1124.md`](file:///deepfreeze/oyahia/BiasEdit/bias_co_ocurrence/CONSIDERATIONS_OLMO2_0425_VS_1124.md)
- **OLMo 2 Technical Report**: [Groeneveld et al., 2025](https://arxiv.org/abs/2501.00656) ([GitHub: allenai/OLMo](https://github.com/allenai/OLMo))
- **Infini-gram Engine**: [Liu et al., 2024](https://arxiv.org/abs/2401.17377) ([GitHub: infinigram/infinigram](https://github.com/infinigram/infinigram))
- **StereoSet Benchmark**: [Nadeem et al., 2021](https://arxiv.org/abs/2004.09456) ([GitHub: moinnadeem/StereoSet](https://github.com/moinnadeem/StereoSet))
