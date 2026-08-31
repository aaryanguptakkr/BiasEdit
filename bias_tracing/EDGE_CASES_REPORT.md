# Which Blank-Locator Should All Models Use? — Findings Report

Audit date: 2026-08-10. Scripts: session scratchpad `partA_legacy_crossval.py`,
`partB_adversarial.py`, `inspect_differs.py`.

## What are we doing and why

Every StereoSet sentence pair has one word that carries the bias
("chief" vs "disorganized"). The code must know **which token positions that word
occupies** — to mark it in plots and analysis. Two methods exist for finding those
positions. This report gives the evidence for deciding which one all models should
use.

## The two methods

**EXISTING method** (currently used by GPT-2 / OLMo / Pythia):

```
sentence:      "My father is chief operator."
make a copy:   "My father is <unk> operator."     <- replace the word with a filler token
tokenize BOTH  -> if same token-length:
find <unk> in the copy -> position 4
therefore "chief" is at position 4 in the real sentence
```

Needs: a filler token, and a hand-written rule per tokenizer family to know
*how many* fillers to insert.

**NEW method** (currently used by Qwen2.5 / Llama-3.2 / Gemma-3; `word_span` in
`dsets/stereoset.py`, after EasyEdit's `repr_tools.py`):

```
tokenize "My father is"        -> 4 tokens     <- word starts at 4
tokenize "My father is chief"  -> 5 tokens     <- word ends at 5
therefore "chief" is at (4,5)
check: do tokens 4-5 of the real sentence spell "chief"?  yes -> accept / no -> discard+log
```

Needs: nothing model-specific. No copy, no filler token, no rules.

## How we verified — three setups

The data: 4,518 templates (gender + profession + race), each with an
anti-stereotype and a stereotype sentence = **9,036 sentences**. Each setup below
checks, for every sentence, the answer to one question: *"which token positions
hold the bias word?"* (e.g. the claim "chief is at tokens 4-5").

**Setup 1 — Self-check (part of the new method itself).**
When the new method claims "the word is at tokens 4-5", it immediately converts
tokens 4-5 of the sentence back into text. If that text is not `"chief"`, the
claim is rejected and the sentence is dropped and its ID logged. This runs inside
the experiment on every sentence it processes — so a wrong position cannot
silently enter any result.

**Setup 2 — Head-to-head on the old models (GPT-2, OLMo-2, Pythia).**
For each of the 9,036 sentences, on each of the 3 old models (= 27,108 checks):
compute the position with the EXISTING method, compute it again with the NEW
method, and compare the two answers.
Result: identical on 25,772. Of the disagreements, every single one traced to a
defect of the EXISTING method (Cases 1-2 below); ~5% of sentences were dropped by
the existing method but located fine by the new one (Case 3); the new method never
dropped a sentence the existing one kept (0 of 27,108).

**Setup 3 — Independent referee on the new models (Qwen, Llama-3, Gemma).**
HuggingFace tokenizers can report, for every token, *which characters of the
original sentence it came from* (`offset_mapping`) — e.g. for
`"My father is chief operator."` it reports that token 4 came from characters
13-18, which is exactly where the letters `chief` sit in the string. This gives an
answer key that involves neither method: we know the word's character positions
just by reading the string, and the offset map tells us which tokens cover those
characters. For each of the 9,036 sentences on each of the 3 new models (= 27,108
checks), the new method's answer was compared against this answer key.
Result: 27,100 exact matches, 0 cases of pointing at a wrong word,
1 unique borderline (Case 5), 0 sentences discarded that the answer key could
locate.

## The cases

### Case 1 — blank is the sentence's first word, model has no BOS (Pythia)
- **Case:** `"Neglect is more of a female trait."` -> tokens `[Neg][lect][ is]...`
- **Correct:** `(0,2)` — both pieces of the word
- **New method:** `(0,2)` OK
- **Existing:** `(1,2)` — cuts off the word's first token (`Neg` lost)
- **Problem:** wrong positions in the published Pythia annotations (17 instances).
- **Reason:** the existing search assumes something always sits at position 0
  (a BOS-like token). Pythia has nothing there except the word itself.
- **What works and why:** the new method makes no position-0 assumption — the
  count of prefix tokens is simply 0 or 1 depending on the model, automatically.

### Case 2 — sentence template has TWO blanks
- **Case:** `"I know a performing artist who is not BLANK and says BLANK."`
- **Correct:** span of the first slot only, e.g. `(9,17)`
- **New method:** `(9,17)` OK
- **Existing:** `(9,27)` — one giant span covering BOTH blanks and everything between
- **Problem:** grossly wrong positions (6 instances: 2 GPT-2, 4 OLMo).
- **Reason:** the copy replaces EVERY "BLANK" with fillers; the search then spans
  from the first filler to the last, unable to tell the two blank sites apart.
- **What works and why:** the new method works from the blank's word position,
  which is unambiguous no matter how many blanks the template has.

### Case 3 — a word the hand-written counting rule mis-counts
- **Case:** any word whose token count in context differs from what the
  per-tokenizer rule predicts (unusual words; whole tokenizer families the rules
  were never written for).
- **Correct:** the word is perfectly locatable.
- **New method:** locates and verifies it. OK
- **Existing:** copy comes out the wrong length -> **sample silently thrown away**.
- **Problem:** invisible data loss — GPT-2 482 instances, OLMo 302, Pythia 511
  (~5% each); routing Gemma through these rules keeps only 13/1026 (1.3%).
- **Reason:** each rule is a human guess about a tokenizer's behavior; every word
  the guess doesn't fit becomes a discarded sample, with no record.
- **What works and why:** the new method doesn't guess — it runs the actual
  tokenizer and subtracts, so the tokenizer's real behavior is in the numbers.

### Case 4 — model has no filler token at all (Qwen2.5, Llama-3.2)
- **Case:** `unk_token = None` in both tokenizers.
- **New method:** unaffected (uses no filler). OK
- **Existing:** crashes (`None * N`).
- **Reason:** the copy needs a real vocabulary token to stand in for the word.
  (Reserved tokens could stand in — but the counting rules would still be wrong
  per Case 3, and with corrected counting the answer already exists before any
  copy is built.)

### Case 5 — the space becomes its own token  **[the new method's ONE flaw found]**
- **Case:** `"Romanians live in rome"` -> `[Roman][ians][ live][ in][ ][rome]`
  (lowercase `rome` has no fused ` rome` vocab entry, so the space stands alone).
- **Correct:** `(5,6)` — just `rome`
- **New method:** `(4,6)` — includes the bare-space token; the self-check's
  `.strip()` forgave the extra space.
- **Existing:** n/a (drops or mangles such samples).
- **Problem:** span one token too wide on the left. Never a wrong word — only
  over-cover. Frequency: 1 unique sentence in 27,108 checks (0.02%); 0 on Gemma.
- **Reason:** the subtraction assigns everything after the prefix to the word;
  normally the space is fused *inside* the word's first token (so that assignment
  is exactly right); here the tokenizer separated them.
- **What works and why:** tolerate (same semantics as the fused-space case) or fix
  with one line — make the self-check exact (`decode == word or " "+word`,
  no strip), which turns this into a logged discard.

### Case 6 — word with internal punctuation (`nurse's`, `co-worker`, garbage fills)
- **Case:** punctuation-stripping turns `nurse's` into `nurses`, which no longer
  matches the sentence text.
- **Both methods:** discard the sample — no wrong span produced (~0.2% of data).
- **Difference:** the new method logs which IDs were discarded; the existing one
  discards silently.

## Why this method and not the alternatives — one example, three ways

Task: in `"My father is chief operator."`, find which token positions hold
**chief**. (A tokenizer chops text into dictionary chunks called tokens;
position = the chunk's index, counting from 0.)

**Approach 1 — hand-write rules for each tokenizer (+ a filler token).**
A human studies each tokenizer and writes a rule for it ("this one glues the
space onto the word, so count ` chief` with the space...", "that one doesn't...").
The rule predicts how many tokens the word takes; a filler copy of the sentence
is then built and searched.
Problem: every rule is a guess about one tokenizer, and every new tokenizer
needs a new guess. We tested the existing rules on the new models — measured,
not argued: Qwen and Llama-3 CRASH outright (they have no filler token at all),
and Gemma silently keeps only 13 of 1026 samples (1.3%) — the other 98.7% are
thrown away without any error message.
→ *Rejected: unmaintainable, and verified to fail badly.*

**Approach 2 — count correctly first, then still use the filler copy.**
Fix Approach 1's guessing by measuring the count instead:

    tokenize("My father is")        = [BOS][My][ father][ is]          → 4 tokens
    tokenize("My father is chief")  = [BOS][My][ father][ is][ chief]  → 5 tokens

So chief occupies exactly 1 token. Then build the copy with a filler and search
for it. Two separate problems:

*(1) It is redundant.* Look at the two numbers you just measured: 4 tokens
before the word means the word STARTS at index 4; 5 tokens through the word
means it ENDS before index 5. The answer — span (4,5) — is already in your
hands. Building a copy to "find" the word answers a question you just answered.

*(2) The copy can still point at the WRONG position — even with a perfect
count.* This is the subtle one. The count answers "HOW MANY tokens is the
word?" (1 — correct). But the copy is used to answer a different question:
"WHERE is it?" — and the copy's own tokenization shifts:

    real: [BOS][My][ father][ is][ chief][ operator][.]      chief at index 4
    copy: [BOS][My][ father][ is][ ][<F>][ operator][.]      filler at index 5!

Why the extra `[ ]`: in the real sentence the space is fused INSIDE `[ chief]`
(spaces always glue forward onto the next word). In the copy, the filler is a
special token that nothing may merge into — so the tokenizer cuts the string at
its boundary, the space loses its partner word, and the orphaned space becomes
its own token. The copy is now one token longer, so the filler sits at index 5
while chief truly sits at index 4. Perfect count, wrong location. (The old code
worked around this by deleting the space together with the word when building
the copy — a trick that is easy to forget; our first attempt forgot it and
mislocated 98% of samples.) Done with full care, this approach agrees with
Approach 3 on 4,092/4,092 tested cases — same answer, longer and breakable road.
→ *Rejected: re-derives an already-known answer through a step that can shift it.*

**Approach 3 — depend on nothing but the tokenizer itself (word_span). CHOSEN.**

*Step 1.* The template says the blank is word #3. Split on spaces:
prefix = `"My father is"`, word = `"chief"` (punctuation stripped).

*Step 2 — start.* `tokenize(prefix)` → 4 tokens. The count of the prefix IS the
index where the next word must sit (positions 0–3 are taken) → start = 4.

*Step 3 — end.* `tokenize(prefix + " chief")` → 5 tokens → end = 5.
Span = (4, 5). A word that splits into pieces (Nurses → [N][urses]) just makes
this count higher — span (4, 6) — with no extra logic.

*Step 4 — why no tokenizer can break the arithmetic.* Every quirk — BOS or no
BOS, space glued or not — appears in BOTH measurements and cancels in the
subtraction. We never need to KNOW the tokenizer's rules: the tokenizer itself
executes them, identically, in both calls.

*Step 5 — "but how do you ENSURE the right tokens were picked?"* The natural
next question — and it has a mechanical answer, not a trust-me answer. Every
span is proof-checked before use: tokenize the full real sentence, decode the
claimed positions back into text, and demand they spell the word:

    decode(tokens[4:5]) = " chief" → strip → "chief" == expected word → ACCEPT
    anything else                                                    → SKIP + log the ID

A wrong span cannot silently enter any result — it either proves itself or the
sample is excluded, visibly. On top of the built-in proof, an independent audit
(the tokenizer's character-offset map, a mechanism the pipeline never uses)
checked all 54,216 spans: zero wrong-word locations, zero wrongly-discarded
samples.

*Step 6 — one bookkeeping detail.* gpt2/olmo/pythia inputs later get a start
marker glued on, shifting every position right by 1; word_span adds that fixed
offset to the returned span (the proof in step 5 still runs in raw coordinates).

No hand-written rules, no filler tokens, identical code for all six models — and
it is the method EasyEdit's ROME implementation adopted for the same reason.

One line: **Approach 1 guesses, Approach 2 measures then re-derives (and can
shift), Approach 3 just measures — and proves every answer.**

## Decision table

| | Existing everywhere | New everywhere |
|---|---|---|
| Cases 1-2 (wrong positions) | keeps the bugs | fixes them (23 spans corrected) |
| Case 3 (silent loss) | loses ~5% | keeps everything + recovers the 5% |
| Cases 4 (new models) | impossible/crash | works (already running, 17/18 runs done) |
| Code | 5 per-tokenizer rule-sets + copy machinery (~100 lines) | one formula + self-check (~10 lines) |
| Cost of switching | — | old-model numbers shift slightly — but only by *additions* and *corrections*; nothing currently kept moves or disappears -> re-run old models, or a one-line disclosure |

**Bottom line:** across 54,216 audited instances the new method produced zero
wrong-word locations and lost zero samples relative to the existing one; its single
blemish is one space-token over-cover with a one-line fix. The existing method
carries two real bug classes (in published old-model annotations) and ~5% silent
data loss. Unifying on the new method can only add data and correct positions.
