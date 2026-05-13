# Bias Tracing Analysis Report
### OLMo-2-0425-1B (Base + Instruct) · Pythia-1B

---

## 1. What We Are Measuring

We want to know: **where inside a language model is bias encoded, and does that location
change across training?**

The tool is **causal tracing** — a method that identifies which specific (layer, token) positions
causally mediate a model's biased prediction. It does not measure whether the model is biased in
general; it localises *where inside the network* the bias signal lives.

---

## 2. The Data

**Dataset:** StereoSet (stereotype consistency subset)

Each data point is a **sentence pair**:
- **Stereotyped sentence:** "The nurse said that *she* was very busy."
- **Anti-stereotyped sentence:** "The nurse said that *he* was very busy."

The **subject** is the word(s) whose identity drives the stereotypic prediction — here, "nurse".

We ask: given the sentence up to the blank, does the model assign higher probability to
the stereotyped completion ("she") than the anti-stereotyped one ("he")?

**Domains and sizes:**

| Domain | Sentence pairs | Notes |
|---|---|---|
| Gender | ~650 | Reliable |
| Race | ~930 | Reliable |
| Profession | ~520 | Reliable |
| Religion | ~30 | **Unreliable** — too few pairs, scores noisy |

---

## 3. The Causal Tracing Process (Step by Step)

For each sentence pair we run three passes through the model:

### Pass 1 — Clean run
Run the model normally on the anti-stereotyped sentence.
Record the probability the model assigns to the stereotyped completion token at each layer.

**`high_score`** = this probability. It is the model's "baseline biased prediction" —
how likely it finds the stereotyped completion when it can read the subject word normally.

### Pass 2 — Corrupted run
Add Gaussian noise to the embedding vectors of the **subject tokens only** ("nurse").
The model can no longer read the subject's identity clearly.

**`low_score`** = the model's probability on the stereotyped completion after this corruption.
This is the floor — the model has lost access to who the subject is.

**`effect_gap`** = `high_score − low_score`
This measures how much the model *relies on the subject's identity* for its biased prediction.
- Large gap (e.g., 0.09) = the subject word is crucial for the bias
- Small gap (< 0.03) = the corruption barely changes the prediction — unreliable signal

### Pass 3 — Restoration (repeated for every layer × token position)
Start from the corrupted run. At one specific (layer L, subject token T), replace the corrupted
hidden state with the clean value from Pass 1. Run the rest of the network normally.

**Indirect effect at (L, T)** = the model's probability on the stereotyped completion after
this single restoration.

A high score here means: *restoring just this one hidden state is enough to substantially
recover the biased prediction.*

### Three restore conditions

Each sentence pair is run three times in Pass 3:
- **States (full restore):** both MLP and Attention outputs at layer L are restored
- **MLP-only:** only the MLP output at layer L is restored; Attention left corrupted
- **Attn-only:** only the Attention output at layer L is restored; MLP left corrupted

This separates which computational pathway (MLP vs Attention) is responsible for carrying the bias.

---

## 4. Score Reference

| Score | What it is |
|---|---|
| `high_score` | Clean-run probability on stereotyped token. How confident the model is when it can see the subject. |
| `low_score` | Corrupted-run probability. Floor — model has lost the subject's identity. |
| `effect_gap` | `high − low`. How much the model depends on the subject. < 0.03 = unreliable. |
| `indirect_effect[L]` | Restoration score at layer L. How much the prediction recovers when only layer L is restored. |
| `states_nie[L]` | Mean indirect effect (States restore) at layer L, averaged over all subject-position tokens and all sentence pairs in the domain. Stored in `stats.json`. |
| `attn_nie[L]` | Same, for Attn-only restore. |
| `mlp_nie[L]` | Same, for MLP-only restore. |

**Important:** `states_nie`, `attn_nie`, `mlp_nie` are raw restoration probabilities — the same
unit as the Y-axis in all bar charts and line plots. They are NOT normalised by the effect gap
unless explicitly stated.

---

## 5. Graph Guide

### 5a. Standard bar chart — `{domain}-states.pdf`

```
plots/OLMo-2-0425-1B/s2-51B/gender-states.pdf
plots/OLMo-2-0425-1B/s2-51B/profession-states.pdf
... (one per checkpoint per domain)
```

**What it shows:** The indirect effect at each layer for one model checkpoint, one domain.

| Axis | Meaning |
|---|---|
| X-axis | Transformer layer (0 = embedding, 15 = final layer for 16-layer models) |
| Y-axis | Indirect effect score (raw restoration probability at subject token positions) |
| Blue bars | States restore — full indirect effect if you give back both MLP + Attn at that layer |
| Red bars | Attn-only — indirect effect through the Attention pathway alone |
| Green bars | MLP-only — indirect effect through the MLP pathway alone |

**How to read it:** A tall blue bar at layer L means: restoring that layer's full hidden state
substantially recovers the biased prediction. The split into red and green tells you whether
that layer's contribution comes from Attention or MLP.

**Key pattern you will see:** Blue bars are highest at L0 and decline toward L15.
Red and green bars are consistently low at all layers.

---

### 5b. Cross-checkpoint line plot — `{domain}-line-checkpoints.pdf`

```
plots/OLMo-2-0425-1B/gender-line-checkpoints.pdf
plots/pythia-1b/race-line-checkpoints.pdf
... (one per model per domain)
```

**What it shows:** How the layer-by-layer indirect effect profile changes across all checkpoints
of one model, for one domain. Three panels side by side (States / Attn-only / MLP-only).

| Axis | Meaning |
|---|---|
| X-axis | Transformer layer (0 to 15) |
| Y-axis | Indirect effect score — same unit as the bar chart |
| Each line | One training checkpoint |
| Line colour | Light = early training, dark = late training (within the same colourmap) |
| Dot on each line | Marks layer 0 (the embedding position) |
| Dashed line | Low-signal checkpoint — effect gap < 0.03, estimates unreliable |

**How to read it:** Watch how the lines shift upward as training progresses (more bias acquired),
and whether the shape (slope from L0 to L15) stays the same or changes.

**Y-axis is shared across all three panels** — you can directly compare the magnitude of
States vs Attn-only vs MLP-only.

---

### 5c. Base vs Instruct line plot — `{domain}-base-vs-instruct-lines.pdf`

```
plots/gender-base-vs-instruct-lines.pdf
plots/profession-base-vs-instruct-lines.pdf
... (one per domain, in plots/ root)
```

**What it shows:** Base model checkpoints (all 8) and Instruct checkpoints (3) on the same axes.

| Element | Meaning |
|---|---|
| Blue solid lines | OLMo-2-0425-1B base checkpoints (light blue = 0B, dark blue = s2-51B) |
| Orange dashed lines | OLMo-2-0425-1B-Instruct checkpoints |
| Circle markers | Layer 0 position on each line |
| Square markers | Layer 0 position on instruct lines |

**How to read it:** If the instruct dashed lines sit above the final base solid line, instruction
tuning increased the bias signal. If the shape (slope) is the same, the causal location did not
change.

---

### 5d. Checkpoint heatmap — `heatmap_checkpoint_layer.pdf`

```
plots/OLMo-2-0425-1B/heatmap_checkpoint_layer.pdf
```

**What it shows:** A 2D grid — checkpoints on X, layers on Y, colour = mean indirect effect.
Two rows: MLP restore (top), Attn restore (bottom). Four columns: the four domains.

**How to read it:** Bright colour at a cell = that (checkpoint, layer) position has high causal
effect. A bright column at early layers across all checkpoints = L0 dominance from the start.

---

## 6. Findings and Observations

### Finding 1 — Bias is lexically encoded at the embedding layer (L0)

**Evidence:** In every domain (except religion which has too few cases), for both OLMo and Pythia,
the highest indirect effect is at L0, and it declines monotonically toward L15.

**Concrete numbers (OLMo base, final checkpoint s2-51B, States restore):**

| Domain | L0 | L4 | L8 | L12 | L15 |
|---|---|---|---|---|---|
| Gender | 0.339 | 0.328 | 0.310 | 0.301 | 0.301 |
| Profession | 0.324 | 0.311 | 0.292 | 0.277 | 0.274 |
| Race | 0.327 | 0.320 | 0.301 | 0.293 | 0.293 |

**Interpretation:** The most causally effective intervention is restoring the token embedding
itself — the vector that represents "nurse" or "father" or "Hispanic". This is because the word's
embedding already encodes stereotypic associations from training data co-occurrences. Words like
"nurse" appeared near "she" orders of magnitude more than near "he" in the training corpus;
that pattern is directly encoded in the embedding vector before any transformer layer runs.

**Why does the score decline across layers?** This is NOT the model becoming less biased at
deeper layers. Bias is present throughout. The score declines because restoring a single hidden
state at a later layer creates an *inconsistent* internal state — that layer's clean hidden state
is surrounded by corrupted context from all other layers and tokens. The inconsistency limits
how much recovery is possible. L0 is the most upstream intervention point; its clean signal
propagates forward through all 15 subsequent layers, giving it maximum leverage.

**Why does this differ from factual recall (ROME paper)?**
For facts like "Eiffel Tower is in Paris," the ROME paper finds the peak causal effect at a
specific mid-layer MLP — the model computes a structured retrieval there. The embedding for
"Eiffel Tower" also encodes Paris associations, but the transformer adds to that signal through
a specific computation. For bias, no such addition happens — the transformer layers propagate
the stereotype without amplifying or concentrating it. The evidence is the monotonically
declining line: if any mid-layer were adding to the bias signal, we would see a bump there.
We do not.

---

### Finding 2 — Bias is acquired very early in training, then consolidated

**Evidence (OLMo base, gender domain, States restore at L0):**

| Checkpoint | Tokens trained | L0 score | Effect gap |
|---|---|---|---|
| 0B | 0 | 0.105 | 0.057 (noisy) |
| 21B | 21 billion | 0.318 | 0.029 |
| 315B | 315 billion | 0.314 | 0.050 |
| 2.4T | 2.4 trillion | 0.320 | 0.068 |
| 4T | 4 trillion | 0.318 | 0.059 |
| s2-3B | stage 2, 3B extra | 0.335 | 0.076 |
| s2-51B | stage 2, 51B extra | 0.339 | 0.075 |

**How to read this table:**
- The raw L0 score jumps from 0.105 to 0.318 between 0B and 21B — a massive early leap.
- After 21B, the raw L0 score barely moves for the rest of training.
- The **effect gap** tells a different story: it shrinks at 21B (0.057 → 0.029), then grows
  steadily through 4T (0.068), then stabilises at stage 2 (~0.075).

**What this means:** The model picks up the linguistic statistics of stereotypic words very fast
(L0 score jumps at 21B). But at 21B it does not yet *depend* on the subject's identity strongly
for its biased predictions — the effect gap is small. As training continues through hundreds of
billions of tokens, the model builds increasing reliance on subject identity (effect gap grows).
By stage 2 the gap has stabilised — stage 2 training did not introduce new bias.

**The L0-dominant, declining shape is set at 21B and never changes.** Training from 21B to 4T
shifts the lines upward in level but does not change their slope or peak location.
This is visible in `plots/OLMo-2-0425-1B/gender-line-checkpoints.pdf` —
the curves are parallel; they do not rotate or develop new peaks.

---

### Finding 3 — Instruction tuning increases bias signal strength but does not change its location

**Evidence (OLMo, profession domain, States restore):**

| Checkpoint | L0 | L8 | L15 | Effect gap |
|---|---|---|---|---|
| Base s2-51B (final) | 0.324 | 0.292 | 0.274 | 0.097 |
| Instruct step200 | 0.369 | 0.323 | 0.305 | 0.119 (+23%) |
| Instruct step1400 | 0.371 | 0.324 | 0.305 | 0.122 (+26%) |
| Instruct step2600 | 0.371 | 0.324 | 0.306 | 0.120 (+24%) |

**Similarly for race:**

| Checkpoint | L0 | L8 | L15 | Effect gap |
|---|---|---|---|---|
| Base s2-51B | 0.327 | 0.301 | 0.293 | 0.094 |
| Instruct step200 | 0.365 | 0.336 | 0.322 | 0.116 (+23%) |
| Instruct step2600 | 0.366 | 0.336 | 0.322 | 0.119 (+26%) |

**Three observations from this table:**

1. **The layer shape is identical.** The slope from L0 to L15 is the same in base and instruct.
   L0 is still the peak; the decline is the same. Instruction tuning did not move the bias to
   different layers and did not create new mid-layer peaks.

2. **The overall level increased.** Every layer's score went up after instruction tuning
   (e.g., profession L0: 0.324 → 0.369). The raw scores are higher because the model is
   generally more confident after fine-tuning (both `high_score` and `low_score` shift up).

3. **The effect gap increased substantially for profession (+24%) and race (+23%).**
   The gap is the clean measure of subject-identity dependence. It increased, meaning the
   instruct model relies *more* on knowing who the subject is for its biased predictions.
   Instruction tuning strengthened the bias signal rather than reducing it.

4. **The jump happened at step200 and then froze.** From step200 to step2600 the gap barely
   moves (profession: 0.119 → 0.122 → 0.120). The instruct training recalibrated the model
   very quickly and then stopped changing bias.

This is visible in `plots/profession-base-vs-instruct-lines.pdf` — the orange dashed
instruct lines sit clearly above the darkest blue base line, with identical slope.

---

### Finding 4 — The same pattern holds across architectures (OLMo vs Pythia)

**OLMo-2-0425-1B:** SwiGLU FFN, rotary position embeddings.
**Pythia-1B:** GPT-NeoX architecture, standard FFN.

Both models show:
- L0 peak, monotonically declining indirect effect
- MLP-only and Attn-only scores flat and low at all layers
- Bias acquired early in training
- Effect gap grows with training

**Notable difference — Pythia race:**

| Checkpoint | L0 | L8 | L15 | Effect gap |
|---|---|---|---|---|
| step5k | 0.311 | 0.302 | 0.300 | 0.032 |
| step81k | 0.300 | 0.269 | 0.263 | 0.103 |
| step143k | 0.298 | 0.267 | 0.261 | 0.112 |

At late Pythia checkpoints, the race L0→L15 slope is steeper (L0=0.298, L15=0.261, drop=0.037)
than OLMo's equivalent (L0=0.327, L15=0.293, drop=0.034 over the same proportion). Also,
Pythia's race effect gap (0.112) at final training is larger than OLMo's (0.094). Whether this
reflects a training-data difference or an architecture difference requires further investigation.

The core finding — L0 dominance, no mid-layer MLP peak — is consistent across both architectures.

---

### Finding 5 — MLP and Attention pathways carry negligible independent bias signal

For both models, at every checkpoint, the MLP-only and Attn-only indirect effect scores are low
and flat across all layers. There is no single layer where either pathway alone is responsible
for carrying the bias.

This is the key contrast with factual recall (ROME): for facts, a specific MLP layer has a sharp
high-indirect-effect peak. The MLP there acts as a key-value store for that fact. For bias, no
such storage site exists in either MLP or Attention pathways.

**Implication:** Editing methods like ROME and MEMIT identify and modify specific MLP layers
based on causal tracing peaks. For bias, there is no such peak to target. These methods attack
a layer structure that does not apply to bias.

---

## 7. What the Findings Mean for Debiasing

The standard debiasing approaches fall into two categories:

**Layer-targeted editing (ROME, MEMIT, MEND):** Find the causally important MLP layer via
causal tracing, then edit the weights there. These work for facts precisely because facts have
a locatable MLP peak. For bias, causal tracing finds no such peak — L0 dominance means the
signal enters through the embedding, not through any MLP computation. Editing mid-layer MLPs
for debiasing is targeting the wrong location.

**Embedding-space approaches:** Methods that operate on the token embedding matrix directly —
such as Hard Debias, INLP, or embedding fine-tuning — are mechanistically better aligned with
where the bias lives. The causal tracing results here provide a mechanistic justification for
why embedding-level interventions are the right target.

**The instruct finding adds another dimension:** Instruction tuning (RLHF/SFT) is the
industry-standard post-training safety technique, yet it increased the effect gap for profession
and race by ~24% rather than reducing it. It did not change the causal location of bias either.
Post-training alignment does not appear to address lexically-encoded stereotypic associations.

---

## 8. File Reference

```
bias_tracing/plots/
│
├── OLMo-2-0425-1B/
│   ├── {domain}-line-checkpoints.pdf     ← Finding 1 + 2: layer profile across all base ckpts
│   ├── {domain}-states-all-checkpoints.pdf  ← Same data as bar charts, one subplot per ckpt
│   ├── heatmap_checkpoint_layer.pdf      ← Overview: all ckpts × all layers, MLP + Attn
│   ├── {domain}-bias-delta.pdf           ← Change per interval (curr − prev checkpoint)
│   ├── report.md                         ← Full numeric tables (NIE per layer, peak layers)
│   └── stats.json                        ← Raw data for all checkpoints and domains
│
├── OLMo-2-0425-1B-Instruct/
│   ├── {domain}-line-checkpoints.pdf     ← Finding 3: instruct layer profiles
│   └── stats.json
│
├── pythia-1b/
│   ├── {domain}-line-checkpoints.pdf     ← Finding 4: cross-architecture comparison
│   └── stats.json
│
├── {domain}-base-vs-instruct-lines.pdf   ← Finding 3: base (solid) vs instruct (dashed), same axes
└── {domain}-bias-trajectory.pdf         ← (supplementary) effect gap trajectory across both phases
```

---

## 9. Terminology Glossary

| Term | Definition |
|---|---|
| **Subject tokens** | The word(s) in the sentence whose identity drives the stereotypic prediction (e.g., "nurse", "father") |
| **Corruption** | Adding Gaussian noise to the subject token embeddings — breaks the model's ability to read who the subject is |
| **Restoration** | Replacing a corrupted hidden state at one (layer, token) position with its clean value |
| **Indirect effect** | The restoration score at a given (layer, token) — how much the prediction recovers |
| **Effect gap** | `high_score − low_score` — how much the model relies on the subject's identity |
| **L0** | Layer 0, the embedding layer — the raw token vector before any transformer computation |
| **States restore** | Full restoration at a layer — both MLP and Attn hidden states replaced |
| **MLP-only restore** | Only the MLP output restored; Attn output left corrupted |
| **Attn-only restore** | Only the Attn output restored; MLP output left corrupted |
| **Low-signal** | Effect gap < 0.03 — corruption barely changes the prediction; indirect effects are unreliable |
