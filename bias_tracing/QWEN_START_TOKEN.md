# Qwen2.5 start-token issue in bias causal tracing

*2026-11-03 · for the decision on how to treat Qwen2.5 in RQ3*

## 1. Summary and decision needed

Qwen2.5 is the only one of our five model families without a start token. In Qwen, the subject is therefore often the very first token, and the corrupted run adds noise to position 0. The other four models never corrupt position 0. This affects 174 of 684 gender, 920 of 1,036 race and 45 of 556 profession samples.

The decision: keep the Qwen results and state the difference in the paper, or rerun Qwen with a start token so that all five model families are treated the same.

## 2. What we do

For each StereoSet sample we run three passes and measure the preference of the model for $x_\text{stereo}$ over $x_\text{anti}$ with the signed log-probability difference

$$\textit{LPD} = \log P_\theta(x_\text{stereo}) - \log P_\theta(x_\text{anti}),$$

where $\log P_\theta(x)$ is the mean log-probability of the tokens of $x$.

1. **Clean run:** the two contexts as they are, giving $\textit{LPD}^\text{clean}$.
2. **Corrupted run:** Gaussian noise is added to the input embeddings of the subject tokens (the bias attribute word, e.g. "Korean"), giving $\textit{LPD}^\text{corrupted}$.
3. **Patched run:** the corrupted run with the clean activation restored at one token position $i$ and layer $\ell$, giving $\textit{LPD}^\text{restored}$.

The effect gap is the mean of $\textit{LPD}^\text{clean} - \textit{LPD}^\text{corrupted}$ over the samples, and NIE is the share of the effect gap that a restored state recovers.

$\log P_\theta(x)$ averages over every token from position 1 onward. Position 0 cannot be scored: a language model predicts each token from the tokens before it, and nothing comes before position 0.

## 3. Start and end tokens in each model

| Model | Start token at position 0 | How it gets there |
| --- | --- | --- |
| OLMo 2 | `<\|endoftext\|>` | added by our code (`make_inputs` in `experiments/bias_trace.py`) |
| Pythia | `<\|endoftext\|>` | added by our code |
| Llama 3.2 | `<\|begin_of_text\|>` | added by its tokenizer |
| Gemma 3 | `<bos>` | added by its tokenizer |
| **Qwen2.5** | **none** | no start token by design |

**Qwen2.5 has end tokens but no start token.** Its tokenizer configuration, in both the Pre and the Post model, sets `bos_token: null` and `add_bos_token: false`. Its end tokens are `<|endoftext|>`, which ends a document and is also Qwen's padding token, and `<|im_end|>`, which ends a chat turn and is the end token of the Instruct model. The Qwen team states that no start token is needed, and TransformerLens also adds none for Qwen.

End tokens play no role in our runs. No model gets an end token appended, because every context ends with its final word or full stop. The only relevance of Qwen's `<|endoftext|>` is that it is the same string we already put at position 0 for OLMo 2 and Pythia, which makes it the natural start token if we rerun Qwen (option C below).

The same race sample, as stored in our result files:

| Model | Position 0 | 1 | 2 | 3 | 4 | Subject positions |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3.2 (OLMo 2, Gemma 3, Pythia alike) | `<\|begin_of_text\|>` | K | orean | people | tend | 1–2 |
| Qwen2.5 | **K** | **orean** | people | tend | to | **0–1** |

In Llama, the noise goes on positions 1–2 and the start token stays clean. In Qwen, the noise goes on positions 0–1, so the first position of the input is corrupted.

Scoring is hardly affected. Qwen's first word gets no probability, but that word is the same in $x_\text{stereo}$ and $x_\text{anti}$, so it would almost cancel in $\textit{LPD}$ anyway.

## 4. The problem

The first position of the input is not an ordinary token. Many attention heads place most of their attention on it whatever it is, using it as a resting place (an "attention sink"; Xiao et al., ICLR 2024). Models also give it unusually large activations (Sun et al., *Massive Activations in Large Language Models*, COLM 2024).

In the four models with a start token, that resting place is the clean start token, so corrupting the subject removes only the information of the subject. In Qwen, when the subject is the first word, the noise also lands on the resting place. The corrupted run may then damage more than the subject information, which changes the baseline that the effect gap and every NIE of Qwen are measured against.

How often the subject is the first token in Qwen:

| Bias type | Samples with the subject first | All samples | Share |
| --- | --- | --- | --- |
| Race | 920 | 1,036 | 89% |
| Gender | 174 | 684 | 25% |
| Profession | 45 | 556 | 8% |

Race is hit hardest because race contexts usually open with the group name, as in "Korean people tend to be…". The same samples and tokenizer are used for Qwen Post and for Pre$\to$Post and Post$\to$Pre, so the same split holds there.

## 5. What the paper claims that involves Qwen2.5

Only claims built on the corrupted run can be affected. Claims that use the clean run alone never touch the noise.

| Claim in the RQ3 section | Qwen2.5 evidence | Uses the corrupted run? |
| --- | --- | --- |
| Post-training keeps bias at the same tokens and layers | The NIE curves of Pre and Post correlate at 0.998–0.999 | Yes |
| Bias first sits at the subject token, then moves to the pre-target token | NIE of the subject's single state at layer 0 is 0.51–0.80; Qwen is in Figure 1 | Yes |
| Which sublayer carries the effect differs between models | In Qwen the MLP window and the Attn window each recover at least 0.46 | Yes |
| Post-training rewrites the representation in some models but not in others | Qwen is the "interchangeable" example in Figure 2: cross-model and within-model NIE differ by at most 0.01 | Yes |
| Post-training separates the two contexts more strongly but does not change the direction | $\overline{\mathrm{abs.}\ \mathit{LPD}}_\text{clean}$ rises in all three Qwen bias types | No |
| Gender carries the strongest stereotyped preference | $\overline{LPD}_\text{clean}$ is highest for gender in Qwen Pre and Post | No |

The cross-model comparison puts Pre and Post under the same corruption, so the problem affects both sides of it equally. What the problem can change are the levels of Qwen's NIE values, race most of all.

## 6. How it impacts our work

The problem lowers the race NIE of Qwen at the subject token and layer 0 by roughly a tenth of the effect gap. It shows no measurable effect on gender, profession or the cross-model result, and none of the claims of the paper changes.

To measure this without rerunning anything, we split the Qwen samples into those with the subject first and those with the subject later. We then compared each group with Llama 3.2 on exactly the same sentences; in Llama the subject is never first, because of its start token. If corrupting position 0 matters, Qwen should fall below Llama only in the "subject first" group. All values below are point estimates from the result files.

| Bias type | Group | Samples | NIE, subject, $\ell=0$: Qwen Pre | NIE, subject, $\ell=0$: Llama Pre | Qwen − Llama | Qwen Pre$\to$Post: cross − within NIE, $\ell=0$ |
| --- | --- | --- | --- | --- | --- | --- |
| Race | Subject first | 920 | 0.50 | 0.62 | −0.12 | +0.004 |
| Race | Subject later | 116 | 0.59 | 0.56 | +0.03 | +0.005 |
| Gender | Subject first | 174 | 0.54 | 0.47 | +0.07 | +0.001 |
| Gender | Subject later | 510 | 0.87 | 0.76 | +0.11 | +0.001 |
| Profession | Subject first | 45 | 0.52 | 0.38 | +0.14 | −0.018 |
| Profession | Subject later | 511 | 0.82 | 0.82 | 0.00 | +0.005 |

- **Race:** Qwen falls 0.12 below Llama only when the subject is first, and matches Llama otherwise. The reported Qwen race value of 0.51 is probably about 0.1 too low because of the problem, but it still lies inside the range of 0.45–0.80 that the paper reports over all 27 pairs.
- **Gender and profession:** there is no drop specific to Qwen. Contexts that open with the subject have a lower subject NIE in both models, so that is a property of the sentences, not of the missing start token.
- **Cross-model patching:** cross-model and within-model NIE differ by at most 0.02 in every group. The conclusion that the activations of Qwen Pre and Post are interchangeable holds whether or not the subject is first.
- **Pre-target token:** its NIE at the last layer is close to 1 in both race groups (0.98 and 0.95), so the pattern from the subject to the pre-target token is unchanged.

Caveats: the two groups contain different sentences, the "subject later" race group has only 116 samples and the profession "subject first" group only 45, and the comparison with Llama is suggestive rather than a controlled test. A rerun with a start token (option C) would give the controlled answer.

## 7. Workarounds

| Option | What it involves | Cost | What it buys |
| --- | --- | --- | --- |
| A. State it in the appendix | One sentence: Qwen2.5 has no start token, so when the subject is the first word, the corruption also falls on the first position | None | Transparency; numbers unchanged |
| B. Add the subset check | Report the comparison in Section 6 in the appendix as a robustness check | Small, CPU only, already computed | Shows that the claims hold in both groups and bounds the effect on race |
| C. Rerun Qwen with a start token | Prepend `<\|endoftext\|>` to the Qwen inputs, as we already do for OLMo 2 and Pythia; rerun Qwen Pre, Post, Pre$\to$Post and Post$\to$Pre for all three bias types and all three patch types; regenerate the Qwen tables, figures and quoted numbers | A code change plus a full GPU rerun of Qwen (12 run sets of 3 patch types); run time to be checked from the earlier Qwen logs | All five model families treated identically; a controlled answer for race |
| D. Drop the Qwen samples with the subject first | Analyse only the "subject later" samples | None | Not advisable: removes 89% of the Qwen race samples and makes the Qwen sample sets differ from the other models |

Option C departs from Qwen's own convention of using no start token. The noise scale $\sigma$ would not change, because it is computed from the subject names alone.

## 8. Recommendation

Do A and B for the submission: state the difference and report the subset check. No claim changes, and the only measurable effect is a lower Qwen race NIE at layer 0 that stays inside the reported range. Keep C as the fallback if reviewers ask for identical treatment, or if there is GPU time before the camera-ready version.
