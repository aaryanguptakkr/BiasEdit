# Plots Guide

## Shared conventions

**Y-axis** — `Abs. log prob diff (stereo − anti)` on all bar/line plots; `Δ` prefix on delta plots.  
**X-axis** — `Layer` (0-indexed transformer layers; L0 = `model.layers.0`, not the raw embedding).  
**⚠** — low-signal checkpoint: `effect_gap < 0.03`, estimates unreliable.

**Bar colors:**

| Color | Label | Meaning |
|---|---|---|
| Blue | Effect of single state | Full restore (MLP + Attn) |
| Red | Effect with Attn severed | MLP-only restore — Attn stays corrupted |
| Green | Effect with MLP severed | Attn-only restore — MLP stays corrupted |

**NIE** = (Patched − Corrupted) / (Clean − Corrupted)  
In code: `(states_nie[L] − mean_low) / effect_gap`.  
Note: `states_nie`, `attn_nie`, `mlp_nie` in `stats.json` store **raw abs log prob diffs**, not normalized NIE.

---

## Files

| File | Type | Notes |
|---|---|---|
| `<model>/{domain}-states-all-checkpoints.pdf` | Bar grid | One subplot per checkpoint, bar colors as above |
| `<model>/{domain}-words-all-checkpoints.pdf` | Bar grid | Same layout; blue = bias words, red = pre-attribute token, green = attribute terms |
| `<model>/{domain}-bias-delta.pdf` | Bar grid | Y = Δ between consecutive checkpoints; zero line drawn |
| `<model>/{domain}-line-checkpoints.pdf` | Line, 3 panels | Light→dark = early→late; panels: States / Attn-only / MLP-only |
| `{domain}-base-vs-instruct.pdf` | Bar grid | `[Base]` vs `[Instruct]` subplots; Y-axis fixed across all |
| `{domain}-base-vs-instruct-lines.pdf` | Line, 3 panels | Solid = base, dashed = instruct; panels: States / Attn-only / MLP-only |
| `{domain}-olmo-vs-pythia.pdf` | Line, 3 panels | Solid blue = OLMo, dashed green = Pythia |
| `{domain}-bias-trajectory.pdf` | Timeline, 3 panels | See below |

**`{domain}-bias-trajectory.pdf` panels** (X = checkpoint, left=base / right=instruct):

| Panel | Y-axis | What it shows |
|---|---|---|
| 1 | Effect gap (high − low) | Overall bias strength per checkpoint |
| 2 | NIE at L0 | Normalized causal signal at first transformer layer |
| 3 | Abs. log prob diff at L0 | Same signal, raw (un-normalized) |
