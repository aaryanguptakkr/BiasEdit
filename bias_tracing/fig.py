"""
Generate all bias-tracing bar-chart plots.

──────────────────────────────────────────────────────────────────────────────
WITHIN-MODEL PATCHING  (--plots bars | delta | compare)
──────────────────────────────────────────────────────────────────────────────
Standard causal tracing: corrupt the subject tokens of one model, restore one
hidden state at a time, measure how much the bias-consistent prediction recovers.

Per checkpoint  →  plots/{model}/{label}/
    {domain}-states.pdf         full / MLP-window / Attn-window restore bars per layer
    {domain}-words.pdf          bias-word / pre-blank / blank-token bars per layer
    composite-states.pdf        all 4 domains side-by-side (states)
    composite-words.pdf         all 4 domains side-by-side (words)
    composite-all.pdf           2×4 grid: top=states, bottom=words

Per domain (across all checkpoints)  →  plots/{model}/
    {domain}-states-all-checkpoints.pdf   one subplot per checkpoint
    {domain}-words-all-checkpoints.pdf
    {domain}-bias-delta.pdf               Δ signal between consecutive checkpoints

Per model  →  plots/{model}/
    stats.json    all numeric data (reloadable without re-running)
    report.md     human-readable summary tables

Data source: shared zip (--source zip, default) or local NFS (--source local).

──────────────────────────────────────────────────────────────────────────────
CROSS-PATCH  (--plots cross_patch)
──────────────────────────────────────────────────────────────────────────────
Cross-model patching: activations from a *source* model are injected into a
*target* model. This tests whether the source model's representations are
sufficient to drive bias predictions in the target — i.e. whether bias is
encoded in a transferable way between pre-training and instruction fine-tuning.

Two directions:
  pre_to_post — source: base pre-trained  /  target: instruct fine-tuned
  post_to_pre — source: instruct fine-tuned  /  target: base pre-trained

Runs exist per model family (CROSS_PATCH_FAMILIES in plot_utils.py:
olmo_1b, qwen2.5_1.5b, llama3.2_1b, gemma3_1b). olmo_1b keeps the original
output paths below; newer families write to a plots/cross_patch/{family}/
subtree with the same file names.

Per direction  →  plots/cross_patch/{direction}/   (olmo_1b)
                  plots/cross_patch/{family}/{direction}/   (other families)
    {domain}-states.pdf         bars per layer (same 3-bar layout as within-model)
    {domain}-words.pdf
    composite-states.pdf        all 4 domains side-by-side
    composite-words.pdf
    composite-all.pdf           2×4 grid

Comparison (both directions)  →  plots/cross_patch/   resp.  plots/cross_patch/{family}/
    {domain}-directions-states.pdf   pre→post vs post→pre, fixed Y-axis
    {domain}-directions-words.pdf

Appendix A4-style 4×3 grid (states/words × domains, both directions) is written
per family: A4-cross-patch-bars.pdf (olmo_1b) / A4-cross-patch-bars-{family}.pdf.

Data source: local filesystem, per-family base dirs (see plot_utils.py).
Use --direction to run only one direction, --cp_family to run only one family.
"""

import os
import json
import zipfile
import argparse
import datetime
import math
import numpy as np
from tqdm import tqdm
import matplotlib
from matplotlib.patches import Patch
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_utils import (
    ZIP_PATH, MAIN_ZIP, PLOTS_BASE,
    MODEL_CONFIGS, BIAS_TYPES, PAPER_DOMAINS,
    CROSS_PATCH_BASE, CROSS_PATCH_CONFIGS, CROSS_PATCH_FAMILIES,
    cross_patch_cases_dir,
    STATES_LABELS, WORDS_LABELS, BAR_COLORS, STATES_COLORS, WORDS_COLORS,
    LOW_SIGNAL, Y_LABEL_BARS, Y_LABEL_NIE,
    FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    FIG_BAR_W_SINGLE, FIG_BAR_H_SINGLE, FIG_BAR_W_PER_COL,
    FIG_GRID_W_PER_COL, FIG_ROW_H, FIG_LINE_W_PER_PAN, FIG_TRAJ_H,
    BASE_COLOR, INSTRUCT_COLOR, LOW_SIG_COLOR, LOW_SIG_BG,
    local_cases_dir, zip_cases_prefix, partition_names, subsample_aligned,
    load_npz_local, load_npz_zip,
    SCORE_METRIC, validate_score_files, collect_scores,
    _draw_bars, _savepdf, normalized_indirect_effect,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

PLOT_CHOICES = ['bars', 'delta', 'compare', 'cross_patch', 'appendix', 'cp_nie']

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=__doc__,
    epilog="""
examples:
  python fig.py                                    # everything (default)
  python fig.py --plots bars                       # only per-checkpoint bar charts
  python fig.py --plots delta compare              # only delta + base-vs-instruct
  python fig.py --plots cross_patch                # only cross-patch plots
  python fig.py --plots cross_patch --direction pre_to_post  # one direction only
  python fig.py --model pythia-1b --plots bars delta
""")
parser.add_argument('--model', default=None, choices=list(MODEL_CONFIGS.keys()),
                    help='run only this model (within-model plots only); omit to run all')
parser.add_argument('--bias', default=None, choices=BIAS_TYPES,
                    help='run only this domain; omit to run all four')
parser.add_argument('--num_sample', type=int, default=None,
                    help='max cases per domain per kind (default: all)')
parser.add_argument('--source', default='zip', choices=['zip', 'local', 'auto'],
                    help='data source for within-model plots: '
                         'zip (recommended on deepfreeze), local (extracted NFS), '
                         'auto (local if extracted, else zip). '
                         'Cross-patch always reads from local filesystem.')
parser.add_argument('--plots', nargs='+', default=['all'],
                    choices=PLOT_CHOICES + ['all'], metavar='PLOT',
                    help=('plots to generate (default: all). '
                          'bars = per-checkpoint bar charts + composites; '
                          'delta = Δ signal between consecutive checkpoints; '
                          'compare = base vs instruct comparison + trajectory; '
                          'cross_patch = cross-model patching plots.'))
parser.add_argument('--direction', default=None,
                    choices=list(CROSS_PATCH_CONFIGS.keys()),
                    help='cross-patch direction to run (default: both). '
                         'Only used when cross_patch is in --plots.')
parser.add_argument('--cp_family', default=None,
                    choices=list(CROSS_PATCH_FAMILIES.keys()),
                    help='cross-patch model family to run (default: all with data). '
                         'Only used when cross_patch/appendix is in --plots.')
args = parser.parse_args()

_plots = set(args.plots)
if 'all' in _plots:
    _plots = set(PLOT_CHOICES)
RUN_BARS        = 'bars'        in _plots
RUN_DELTA       = 'delta'       in _plots
RUN_COMPARE     = 'compare'     in _plots
RUN_CROSS_PATCH = 'cross_patch' in _plots
RUN_APPENDIX    = 'appendix'    in _plots
RUN_CP_NIE      = 'cp_nie'      in _plots

models_to_run      = [args.model]     if args.model     else list(MODEL_CONFIGS.keys())
domains_to_run     = [args.bias]      if args.bias       else BIAS_TYPES
directions_to_run  = [args.direction] if args.direction  else list(CROSS_PATCH_CONFIGS.keys())
cp_families_to_run = [args.cp_family] if args.cp_family  else list(CROSS_PATCH_FAMILIES.keys())

# ── data helpers ──────────────────────────────────────────────────────────────

def save_individual(r1, r2, r3, labels, colors, num_layer, title, savepath):
    fig, ax = plt.subplots(figsize=(FIG_BAR_W_SINGLE, FIG_BAR_H_SINGLE))
    _draw_bars(ax, r1, r2, r3, labels, colors, num_layer,
               'Layer', Y_LABEL_BARS, title)
    plt.tight_layout()
    _savepdf(fig, savepath)

def save_composite(domain_data, plot_type, model_name, ckpt_label, out_dir):
    """1×N grid — all domains side-by-side for one checkpoint."""
    domains = [d for d in BIAS_TYPES if d in domain_data]
    if not domains:
        return
    labels       = STATES_LABELS if plot_type == 'states' else WORDS_LABELS
    title_suffix = 'effect of states' if plot_type == 'states' else 'effect of different words'
    fig, axes = plt.subplots(1, len(domains),
                             figsize=(FIG_BAR_W_PER_COL * len(domains), FIG_ROW_H))
    if len(domains) == 1:
        axes = [axes]
    fig.suptitle(f'Bias {title_suffix} — {model_name}  [{ckpt_label}]',
                 fontsize=FS_SUPTITLE, fontweight='bold')
    for ax, domain in zip(axes, domains):
        r1, r2, r3, nl = domain_data[domain]
        _draw_bars(ax, r1, r2, r3, labels, BAR_COLORS, nl,
                   'Layer', Y_LABEL_BARS, domain.title())
    plt.tight_layout()
    _savepdf(fig, os.path.join(out_dir, f'composite-{plot_type}.pdf'))

def save_composite_all(states_data, words_data, model_name, ckpt_label, out_dir):
    """2×N grid — top row=states, bottom row=words, columns=domains."""
    domains = [d for d in BIAS_TYPES if d in states_data and d in words_data]
    if not domains:
        return
    fig, axes = plt.subplots(2, len(domains),
                             figsize=(FIG_BAR_W_PER_COL * len(domains), FIG_ROW_H * 2))
    fig.suptitle(f'All bias domains — {model_name}  [{ckpt_label}]',
                 fontsize=FS_SUPTITLE, fontweight='bold')
    for col, domain in enumerate(domains):
        r1, r2, r3, nl = states_data[domain]
        _draw_bars(axes[0, col], r1, r2, r3, STATES_LABELS, BAR_COLORS, nl,
                   'Layer', Y_LABEL_BARS, f'{domain.title()} — states')
        r1, r2, r3, nl = words_data[domain]
        _draw_bars(axes[1, col], r1, r2, r3, WORDS_LABELS, BAR_COLORS, nl,
                   'Layer', Y_LABEL_BARS, f'{domain.title()} — words')
    plt.tight_layout()
    _savepdf(fig, os.path.join(out_dir, 'composite-all.pdf'))

def save_cross_checkpoint(domain_arrays, plot_type, model_name, domain, out_dir):
    """
    Grid of bar charts — one subplot per checkpoint, 2 columns.
    domain_arrays: ordered list of (ckpt_label, r1, r2, r3, num_layer)
    """
    if not domain_arrays:
        return
    n      = len(domain_arrays)
    ncols  = 2
    nrows  = math.ceil(n / ncols)
    labels = STATES_LABELS if plot_type == 'states' else WORDS_LABELS
    title_suffix = 'effect of states' if plot_type == 'states' else 'effect of different words'

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIG_GRID_W_PER_COL * ncols, FIG_ROW_H * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]
    fig.suptitle(f'{domain.title()} bias {title_suffix} — {model_name} (all checkpoints)',
                 fontsize=FS_SUPTITLE, fontweight='bold')

    for i, (ckpt_label, r1, r2, r3, nl) in enumerate(domain_arrays):
        _draw_bars(axes_flat[i], r1, r2, r3, labels, BAR_COLORS, nl,
                   'Layer', Y_LABEL_BARS, ckpt_label)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _savepdf(fig, os.path.join(out_dir, f'{domain}-{plot_type}-all-checkpoints.pdf'))

# ── report helpers ────────────────────────────────────────────────────────────

def _top_layers(scores, n=3):
    return np.argsort(scores)[::-1][:n].tolist()

def save_stats_and_report(model_name, all_ckpt_stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # stats.json — full numeric data, reloadable without re-running
    json_path = os.path.join(out_dir, 'stats.json')
    with open(json_path, 'w') as f:
        json.dump({
            'model':       model_name,
            'generated':   datetime.date.today().isoformat(),
            'num_sample':  args.num_sample,
            'stats_schema': 'raw_patch_scores_v2',
            'score_metric': SCORE_METRIC,
            'checkpoints': all_ckpt_stats,
        }, f, indent=2)
    print(f'  Saved: {json_path}')

    # report.md — human-readable summary
    md_path = os.path.join(out_dir, 'report.md')
    today   = datetime.date.today().isoformat()
    lines   = [
        f'# {model_name} — Bias Tracing Report',
        f'',
        f'Generated: {today}  ',
        f'',
        f'## What this report measures',
        f'',
        f'Causal tracing asks: *which (subject token, layer) positions causally mediate bias?*',
        f'',
        f'For each sentence pair (stereotyped vs. anti-stereotyped), the subject tokens are corrupted',
        f'with Gaussian noise. Then, one hidden state at a time is restored to its clean value.',
        f'The **indirect effect** at (token i, layer j) = how much the prediction recovers when',
        f'only that one state is restored. Reported here as NIE (normalized by the clean–corrupted gap).',
        f'',
        f'Three restore conditions per sentence pair:',
        f'- **Full restore** (single state): all components (MLP + Attn) restored at that layer',
        f'- **MLP-only**: only MLP output restored; Attn output left corrupted',
        f'- **Attn-only**: only Attn output restored; MLP output left corrupted',
        f'',
        f'Scores are aggregated over **subject token positions only** (not the full sentence),',
        f'then averaged over all sentence pairs in the domain.',
        f'',
        f'### Interpreting the NIE pattern',
        f'',
        f'In factual recall (ROME paper), NIE peaks sharply at **specific mid-layer MLPs** — the',
        f'knowledge is "stored" there and computed on demand. Bias may behave differently:',
        f'',
        f'- **NIE highest at L0 and declining**: bias is primarily lexical — it enters through',
        f'  the token embedding and is not further computed or concentrated by transformer layers.',
        f'  Words like "father" or "Hispanic" carry the stereotypic signal in their embedding itself.',
        f'- **NIE goes negative at later layers**: restoring a subject state at a late layer',
        f'  creates an *inconsistent* internal state (one clean token among corrupted context),',
        f'  which can hurt prediction below the corrupted baseline.',
        f'- **MLP-only NIE flat or negative**: the MLP pathway alone does not localize bias,',
        f'  unlike factual knowledge where a specific MLP layer is the key mediator.',
        f'',
        f'⚠ **Religion domain**: very few cases (24–44) and often tiny effect gap (< 0.03).',
        f'NIE estimates for religion are unreliable — treat with caution.',
        f'',
        f'---',
        f'',
        f'## Field reference',
        f'',
        f'| Field | Description |',
        f'|---|---|',
        f'| **N cases** | Sentence pairs processed |',
        f'| **High score** | Mean clean-run absolute whole-sentence log-prob gap |',
        f'| **Low score** | Mean corrupted-run absolute whole-sentence log-prob gap |',
        f'| **Effect gap** | High − Low — reduction in absolute separation after corruption; < 0.03 = low-signal |',
        f'| **Peak All/MLP/Attn** | Layer with highest raw patched score under each restore condition |',
        f'| **NIE L0** | Normalized indirect effect at the embedding layer |',
        f'| **NIE L-mid** | NIE at the middle layer |',
        f'| **NIE L-last** | NIE at the final layer |',
        f'',
        f'---',
        f'',
        f'## Summary table',
        f'',
        f'| Checkpoint | Domain | N | Gap | Peak All | Peak MLP | Peak Attn | NIE L0 | NIE L-mid | NIE L-last |',
        f'|---|---|---|---|---|---|---|---|---|---|',
    ]
    for e in all_ckpt_stats:
        for domain in BIAS_TYPES:
            s = e['domains'].get(domain)
            if s is None:
                lines.append(f'| {e["label"]} | {domain} | — | — | — | — | — | — | — | — |')
                continue
            gap  = s['effect_gap']
            low  = s['mean_low']
            flag = ' ⚠' if gap < LOW_SIGNAL else ''
            nl   = s['num_layers']
            mid  = nl // 2
            nie_l0   = normalized_indirect_effect(s['states_score'][0],   low, gap, degenerate='zero')
            nie_lmid = normalized_indirect_effect(s['states_score'][mid], low, gap, degenerate='zero')
            nie_last = normalized_indirect_effect(s['states_score'][-1],  low, gap, degenerate='zero')
            lines.append(
                f"| {e['label']} | {domain}{flag} | {s['n_cases']} | {gap:.4f} "
                f"| {s['peak_layer_states']} | {s['peak_layer_mlp']} | {s['peak_layer_attn']} "
                f"| {nie_l0:+.2f} | {nie_lmid:+.2f} | {nie_last:+.2f} |"
            )

    lines += [
        f'',
        f'---',
        f'',
        f'## Normalized Indirect Effect (NIE) by layer — States (full restore)',
        f'',
        f'NIE = (restoration_score - low_score) / (high_score - low_score).',
        f'',
        f'- **NIE > 0**: restoring this (subject token, layer) recovers some of the clean prediction.',
        f'- **NIE = 1**: full recovery to clean-run probability.',
        f'- **NIE < 0**: restoring this position makes prediction *worse* than the corrupted baseline — '
        f'the model\'s internal state has become inconsistent from partially restoring only one position.',
        f'',
        f'⚠ Rows marked `[low-signal]` have gap < 0.03 — too small for reliable NIE estimates.',
        f'',
    ]
    for e in all_ckpt_stats:
        nl = e['domains'][next(iter(e['domains']))]['num_layers'] if e['domains'] else 0
        hdr = '| Domain | ' + ' | '.join(f'L{i}' for i in range(nl)) + ' |'
        sep = '|---|' + '---|' * nl
        lines += [f'### {e["label"]}  `{e["checkpoint"]}`', '', hdr, sep]
        for domain in BIAS_TYPES:
            s = e['domains'].get(domain)
            if s is None:
                lines.append(f'| {domain} | ' + ' | '.join(['—'] * nl) + ' |')
            else:
                gap  = s['effect_gap']
                low  = s['mean_low']
                flag = '  ⚠ low-signal' if gap < LOW_SIGNAL else ''
                nie_vals = normalized_indirect_effect(s['states_score'], low, gap, degenerate='zero')
                vals = ' | '.join(f'{v:+.2f}' for v in nie_vals)
                lines.append(f'| {domain}{flag} | {vals} |')
        lines.append('')

    lines += [
        '---',
        '',
        '## Output files',
        '',
        '```',
        f'plots/{model_name}/',
        '├── stats.json                          ← full numeric data (reload without re-running)',
        '├── report.md                           ← this file',
        '├── heatmap_checkpoint_layer.pdf        ← checkpoint × layer heatmap (MLP + Attn)',
        '├── {domain}-states-all-checkpoints.pdf ← all checkpoints in one figure (per domain)',
        '├── {domain}-words-all-checkpoints.pdf',
        '└── {label}/                            ← one folder per checkpoint',
        '    ├── {domain}-states.pdf',
        '    ├── {domain}-words.pdf',
        '    ├── composite-states.pdf',
        '    ├── composite-words.pdf',
        '    └── composite-all.pdf',
        '```',
    ]

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved: {md_path}')

# ── cross-model plot helpers ──────────────────────────────────────────────────

DOMAIN_COLORS = {'gender': '#2196F3', 'profession': '#4CAF50',
                 'race': '#FF5722', 'religion': '#9C27B0'}

def save_bias_delta(ckpt_stats_list, model_name, out_dir):
    """
    One PDF per domain: same 3-bar-per-layer layout as the standard bar charts,
    but Y-axis shows Δ abs. log prob diff (curr − prev checkpoint) for each layer.
    One subplot per consecutive checkpoint pair, labeled 'prev → curr'.
    """
    DELTA_LABELS = [
        'Δ Effect of single state',
        'Δ Effect of MLP window restore',
        'Δ Effect of Attn window restore',
    ]

    for domain in BIAS_TYPES:
        pairs = []   # list of (pair_label, d_states, d_attn_sev, d_mlp_sev, num_layer, low_signal)
        for i in range(1, len(ckpt_stats_list)):
            prev = ckpt_stats_list[i - 1]
            curr = ckpt_stats_list[i]
            sp   = prev['domains'].get(domain)
            sc   = curr['domains'].get(domain)
            if sp is None or sc is None:
                continue

            pair_label = f'{prev["label"]} → {curr["label"]}'
            nl         = sc['num_layers']
            gap_p, gap_c = sp['effect_gap'], sc['effect_gap']

            def raw_arr(ckpt_s):
                # Return (states, mlp_only, attn_only) — order matches bar chart convention:
                # 2nd bar (red)   = restored window of MLP outputs  (_mlp.npz)
                # 3rd bar (green) = restored window of Attn outputs (_attn.npz)
                return (np.array(ckpt_s['states_score']),
                        np.array(ckpt_s['mlp_score']),
                        np.array(ckpt_s['attn_score']))

            s_p, a_p, m_p = raw_arr(sp)
            s_c, a_c, m_c = raw_arr(sc)

            low_sig = gap_p < LOW_SIGNAL or gap_c < LOW_SIGNAL
            pairs.append((pair_label, s_c - s_p, a_c - a_p, m_c - m_p, nl, low_sig))

        if not pairs:
            continue

        # global y-limits across all intervals so subplots are directly comparable
        all_deltas = np.concatenate([
            np.concatenate([ds, da, dm])
            for _, ds, da, dm, _, _ in pairs
        ])
        global_margin = (np.nanmax(all_deltas) - np.nanmin(all_deltas)) * 0.12 or 0.05
        y_min = np.nanmin(all_deltas) - global_margin
        y_max = np.nanmax(all_deltas) + global_margin

        n     = len(pairs)
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(FIG_GRID_W_PER_COL * ncols, FIG_ROW_H * nrows),
                                 constrained_layout=True)
        axes_flat = axes.flatten() if n > 1 else [axes]

        fig.suptitle(
            f'{model_name} — {domain.capitalize()} bias: Δ per layer vs. previous checkpoint\n'
            'Bars show change in abs. log prob diff (curr − prev) at each layer. '
            'Positive = more causal effect acquired.  Y-axis fixed across all intervals.',
            fontsize=FS_SUPTITLE, fontweight='bold')

        for ax, (pair_label, ds, da, dm, nl, low_sig) in zip(axes_flat, pairs):
            _draw_bars(ax, ds, da, dm, DELTA_LABELS, BAR_COLORS, nl,
                       'Layer', 'Δ Abs. log prob diff (stereo − anti)', pair_label)
            ax.set_ylim(y_min, y_max)   # override per-subplot limits
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            if low_sig:
                ax.set_facecolor(LOW_SIG_BG)
                ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                        fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        out = os.path.join(out_dir, f'{domain}-bias-delta.pdf')
        _savepdf(fig, out)


def save_bias_trajectory(base_stats, instruct_stats, out_dir):
    """
    How bias strength evolves across both training phases on a single timeline.
    Three panels:
      1. Effect gap (high − low) — overall bias strength, raw abs log prob diff units.
      2. Embedding layer contribution — fraction of effect gap recovered at L0
         = (states_score[0] − mean_low) / effect_gap. Scale-invariant across checkpoints.
      3. Raw abs. log prob diff at L0 (states_score[0]) — absolute causal signal at first transformer layer.
    Base checkpoints appear on the left; instruct fine-tuning on the right;
    a vertical dashed line marks the phase boundary.
    Output: {domain}-bias-trajectory.pdf  (one per domain)
    """
    for domain in BIAS_TYPES:
        base_pts, instruct_pts = [], []
        for e in base_stats:
            s = e['domains'].get(domain)
            if not s:
                continue
            gap    = s['effect_gap']
            raw_l0 = s['states_score'][0]
            frac_l0 = normalized_indirect_effect(raw_l0, s['mean_low'], gap, degenerate='nan')
            base_pts.append((e['label'], gap, frac_l0, raw_l0, gap < LOW_SIGNAL))

        for e in instruct_stats:
            s = e['domains'].get(domain)
            if not s:
                continue
            gap    = s['effect_gap']
            raw_l0 = s['states_score'][0]
            frac_l0 = normalized_indirect_effect(raw_l0, s['mean_low'], gap, degenerate='nan')
            instruct_pts.append((e['label'], gap, frac_l0, raw_l0, gap < LOW_SIGNAL))

        if not base_pts and not instruct_pts:
            continue

        n_base  = len(base_pts)
        all_pts = base_pts + instruct_pts
        n_total = len(all_pts)
        labels   = [p[0] for p in all_pts]
        gaps     = np.array([p[1] for p in all_pts], dtype=float)
        frac_l0s = np.array([p[2] for p in all_pts], dtype=float)
        raw_l0s  = np.array([p[3] for p in all_pts], dtype=float)
        low_sig  = [p[4] for p in all_pts]
        xs       = np.arange(n_total)

        fig, (ax_gap, ax_frac, ax_raw) = plt.subplots(
            3, 1, figsize=(max(10, n_total * 1.4), FIG_TRAJ_H),
            constrained_layout=True)

        fig.suptitle(
            f'OLMo-2-0425-1B — {domain.capitalize()} bias: learning trajectory\n'
            'Left: base pre-training checkpoints   |   Right: instruction fine-tuning',
            fontsize=FS_SUPTITLE, fontweight='bold')

        panel_specs = [
            (ax_gap, gaps,
             'Effect gap (high − low)',
             'Effect gap — how much corrupting subject tokens reduces bias-consistent probability\n'
             'Larger = model relies more on subject identity for this domain'),
            (ax_frac, frac_l0s,
             'NIE at L0',
             'NIE at L0 — normalized indirect effect at the first transformer layer\n'
             '= (Patched − Corrupted) / (Clean − Corrupted)'),
            (ax_raw, raw_l0s,
             'Abs. log prob diff at L0\n(stereo − anti)',
             'Raw causal signal at the first transformer layer — absolute scale\n'
             'Reflects both L0 importance and overall bias strength'),
        ]

        for ax, vals, ylabel, title in panel_specs:
            ax.plot(xs[:n_base], vals[:n_base], 'o-',
                    color=BASE_COLOR, label='Base (pre-training)',
                    linewidth=2, markersize=7, zorder=3)
            if instruct_pts:
                bridge_xs = [n_base - 1] + list(xs[n_base:])
                bridge_ys = [vals[n_base - 1]] + list(vals[n_base:])
                ax.plot(bridge_xs, bridge_ys, 's--',
                        color=INSTRUCT_COLOR, label='Instruct fine-tuning',
                        linewidth=2, markersize=7, zorder=3)
                ax.axvline(n_base - 0.5, color='gray', linestyle='--',
                           linewidth=1.2, alpha=0.6)

            for xi, ls in enumerate(low_sig):
                if ls:
                    ax.annotate('⚠', (xs[xi], vals[xi]),
                                textcoords='offset points', xytext=(0, 6),
                                ha='center', fontsize=FS_LABEL, color=LOW_SIG_COLOR)

            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=FS_TICK)
            ax.set_ylabel(ylabel, fontsize=FS_LABEL)
            ax.set_title(title, fontsize=FS_TITLE)
            ax.legend(fontsize=FS_LEGEND)
            ax.grid(axis='y', alpha=0.25)
            ax.axhline(0, color='black', linewidth=0.7, alpha=0.3)

        out = os.path.join(out_dir, f'{domain}-bias-trajectory.pdf')
        _savepdf(fig, out)


def save_base_vs_instruct(base_stats, instruct_stats, out_dir):
    """
    One PDF per domain. Same 3-bar-per-layer layout as the standard bar charts,
    showing full NIE profile (all layers) for:
      - the base model's final checkpoint
      - all instruct fine-tuning checkpoints
    Subplots arranged in 2 columns, labeled '[Base]' or '[Instruct]'.
    Y-axis fixed across all subplots for direct comparison.
    """
    def raw_arrays(s):
        """Return (states, mlp_only, attn_only) raw log prob diff arrays.
        Order matches bar chart convention:
          2nd bar (red)   = restored window of MLP outputs  (_mlp.npz)
          3rd bar (green) = restored window of Attn outputs (_attn.npz)
        """
        return (np.array(s['states_score']),
                np.array(s['mlp_score']),
                np.array(s['attn_score']))

    for domain in BIAS_TYPES:
        subplots = []

        # last base checkpoint
        last_base = base_stats[-1]
        sb = last_base['domains'].get(domain)
        if sb:
            s, a, m = raw_arrays(sb)
            subplots.append((
                f'[Base] {last_base["label"]}',
                s, a, m, sb['num_layers'],
                sb['effect_gap'] < LOW_SIGNAL,
            ))

        # all instruct checkpoints
        for e in instruct_stats:
            si = e['domains'].get(domain)
            if si:
                s, a, m = raw_arrays(si)
                subplots.append((
                    f'[Instruct] {e["label"]}',
                    s, a, m, si['num_layers'],
                    si['effect_gap'] < LOW_SIGNAL,
                ))

        if not subplots:
            continue

        # shared y-limits across all subplots
        all_vals = np.concatenate([
            np.concatenate([s, a, m]) for _, s, a, m, _, _ in subplots
        ])
        margin = (np.nanmax(all_vals) - np.nanmin(all_vals)) * 0.12 or 0.05
        y_min, y_max = np.nanmin(all_vals) - margin, np.nanmax(all_vals) + margin

        n     = len(subplots)
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(FIG_GRID_W_PER_COL * ncols, FIG_ROW_H * nrows),
                                 constrained_layout=True)
        axes_flat = axes.flatten() if n > 1 else [axes]

        fig.suptitle(
            f'OLMo-2-0425-1B — {domain.capitalize()} bias: Base vs. Instruct\n'
            'Abs. log prob diff per layer  (blue = States, red = MLP window, green = Attn window)  '
            'Y-axis fixed across all subplots.',
            fontsize=FS_SUPTITLE, fontweight='bold')

        for ax, (title, s, a, m, nl, low_sig) in zip(axes_flat, subplots):
            _draw_bars(ax, s, a, m, STATES_LABELS, BAR_COLORS, nl,
                       'Layer', Y_LABEL_BARS, title)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            if low_sig:
                ax.set_facecolor(LOW_SIG_BG)
                ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                        fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        out = os.path.join(out_dir, f'{domain}-base-vs-instruct.pdf')
        _savepdf(fig, out)


# ── cross-patch functions ─────────────────────────────────────────────────────
#
# Cross-model patching injects activations from a *source* model into a *target*
# model at one (token, layer) position at a time and measures how much the
# prediction recovers toward the source model's bias-consistent output.
#
# Data format is identical to within-model causal tracing (.npz files with the
# same keys: scores, corrupt_range_anti, blank_idxs_anti, high_score, low_score).
# File naming uses the same _{attn,mlp}.npz suffix convention, so partition_names
# and collect_scores work without modification.
#
# Data lives at {family base}/{family}_{direction}/{domain}/causal_trace/cases/
# (local filesystem, not in the zip — always read with load_npz_local).


def _validate_result_group(file_lists, loader, expected, label):
    try:
        validate_score_files(file_lists, loader, expected)
        return True
    except ValueError as ex:
        print(f'    [{label}] Invalid result set: {ex}')
        return False


def load_cross_patch_domain(direction_key, domain, num_sample=None, family='olmo_1b'):
    """
    Load cross-patch .npz files for one (family, direction, domain) triple.

    direction_key : key in CROSS_PATCH_CONFIGS ('pre_to_post' or 'post_to_pre')
    domain        : one of BIAS_TYPES
    num_sample    : max files per kind (None = all)
    family        : key in CROSS_PATCH_FAMILIES; defaults to 'olmo_1b' so all
                    pre-existing call sites keep their original behavior

    Returns a result dict or None if no data is available:
      bias_mean, pre_blank_mean, blank_mean  — (n_layers,) subject / word arrays
      attn_mean, mlp_mean                    — Attn-only / MLP-only restore arrays
      n_cases, mean_high, mean_low           — scalar summary stats
      effect_gap                             — mean_high − mean_low
      low_sig                                — True if effect_gap < LOW_SIGNAL
      num_layer                              — number of transformer layers
    """
    cases_dir = cross_patch_cases_dir(family, direction_key, domain)

    if not os.path.isdir(cases_dir):
        print(f'    [cross_patch] No directory: {cases_dir}')
        return None

    all_files = sorted(os.listdir(cases_dir))
    single_b, attn_b, mlp_b = partition_names(all_files)
    single_b, attn_b, mlp_b = subsample_aligned(single_b, attn_b, mlp_b, num_sample)

    single_items = [os.path.join(cases_dir, b) for b in single_b]
    attn_items   = [os.path.join(cases_dir, b) for b in attn_b]
    mlp_items    = [os.path.join(cases_dir, b) for b in mlp_b]

    print(f'    single={len(single_items)}, attn={len(attn_items)}, mlp={len(mlp_items)}')

    if not single_items:
        print('    No single-state files; skipping.')
        return None

    family_cfg = CROSS_PATCH_FAMILIES[family]
    if direction_key == 'pre_to_post':
        source_model = family_cfg['base_model']
        target_model = family_cfg['instruct_model']
    else:
        source_model = family_cfg['instruct_model']
        target_model = family_cfg['base_model']
    expected = {
        'score_metric': SCORE_METRIC,
        'source_model': source_model,
        'target_model': target_model,
        'direction': f'{source_model} -> {target_model}',
    }
    if not _validate_result_group(
            (single_items, attn_items, mlp_items), load_npz_local,
            expected, 'cross_patch'):
        return None

    try:
        num_layer = load_npz_local(single_items[0])['scores'].shape[-1]
    except Exception as ex:
        print(f'    Cannot read sample: {ex}; skipping.')
        return None

    bias_mean, pre_blank_mean, blank_mean, n_cases, mean_high, mean_low = \
        collect_scores(single_items, load_npz_local)
    attn_mean, _, _, _, _, _ = collect_scores(attn_items, load_npz_local)
    mlp_mean,  _, _, _, _, _ = collect_scores(mlp_items,  load_npz_local)

    if bias_mean is None:
        print('    No valid scores; skipping.')
        return None

    zero           = np.zeros(num_layer)
    attn_mean      = attn_mean      if attn_mean      is not None else zero
    mlp_mean       = mlp_mean       if mlp_mean       is not None else zero
    pre_blank_mean = pre_blank_mean if pre_blank_mean is not None else zero
    blank_mean     = blank_mean     if blank_mean     is not None else zero

    effect_gap = mean_high - mean_low
    return {
        'bias_mean':      bias_mean,
        'pre_blank_mean': pre_blank_mean,
        'blank_mean':     blank_mean,
        'attn_mean':      attn_mean,
        'mlp_mean':       mlp_mean,
        'n_cases':        n_cases,
        'mean_high':      mean_high,
        'mean_low':       mean_low,
        'effect_gap':     effect_gap,
        'low_sig':        effect_gap < LOW_SIGNAL,
        'num_layer':      num_layer,
    }


def save_cross_patch_direction(direction_key, domain_results, out_dir, family_label=''):
    """
    Per-direction individual PDFs + composite figures.

    domain_results : {domain: result_dict from load_cross_patch_domain}
    out_dir        : plots/cross_patch/{family}/{direction_key}/
    family_label   : display name of the model family, prepended to titles

    Mirrors the per-checkpoint layout used for within-model tracing:
      {domain}-states.pdf      full / MLP-window / Attn-window restore bars per layer
      {domain}-words.pdf       bias-word / pre-blank / blank-token bars per layer
      composite-states.pdf     all 4 domains side-by-side
      composite-words.pdf
      composite-all.pdf        2×N grid: top=states, bottom=words
    """
    cfg   = CROSS_PATCH_CONFIGS[direction_key]
    label = f'{family_label} {cfg["label"]}'.strip()
    desc  = cfg['desc']

    def _low_sig_decorate(ax, res):
        """Add ⚠ annotation and background tint for low-signal panels."""
        if res['low_sig']:
            ax.set_facecolor(LOW_SIG_BG)
            ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                    fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

    # ── individual PDFs per domain ────────────────────────────────────────────
    for domain, res in domain_results.items():
        nl       = res['num_layer']
        sig_flag = ' [low-signal]' if res['low_sig'] else ''

        for plot_type, r2, r3, labels_list in [
            ('states', res['mlp_mean'],       res['attn_mean'],  STATES_LABELS),
            ('words',  res['pre_blank_mean'], res['blank_mean'], WORDS_LABELS),
        ]:
            fig, ax = plt.subplots(figsize=(FIG_BAR_W_SINGLE, FIG_BAR_H_SINGLE))
            _draw_bars(ax, res['bias_mean'], r2, r3, labels_list, BAR_COLORS, nl,
                       'Layer', Y_LABEL_BARS,
                       f'{domain.title()} — {label}{sig_flag}\n{desc}')
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            _low_sig_decorate(ax, res)
            plt.tight_layout()
            _savepdf(fig, os.path.join(out_dir, f'{domain}-{plot_type}.pdf'))

    # ── composite figures ─────────────────────────────────────────────────────
    if not domain_results:
        return

    domains = [d for d in BIAS_TYPES if d in domain_results]

    for plot_type, labels_list in [('states', STATES_LABELS), ('words', WORDS_LABELS)]:
        fig, axes = plt.subplots(1, len(domains),
                                 figsize=(FIG_BAR_W_PER_COL * len(domains), FIG_ROW_H))
        if len(domains) == 1:
            axes = [axes]
        fig.suptitle(f'Cross-patch {label} — {plot_type.title()}\n{desc}',
                     fontsize=FS_SUPTITLE, fontweight='bold')
        for ax, domain in zip(axes, domains):
            res = domain_results[domain]
            if plot_type == 'states':
                r2, r3 = res['mlp_mean'], res['attn_mean']
            else:
                r2, r3 = res['pre_blank_mean'], res['blank_mean']
            _draw_bars(ax, res['bias_mean'], r2, r3, labels_list, BAR_COLORS, res['num_layer'],
                       'Layer', Y_LABEL_BARS, domain.title())
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            _low_sig_decorate(ax, res)
        plt.tight_layout()
        _savepdf(fig, os.path.join(out_dir, f'composite-{plot_type}.pdf'))

    # 2×N grid: top row = states, bottom row = words
    fig, axes = plt.subplots(2, len(domains),
                             figsize=(FIG_BAR_W_PER_COL * len(domains), FIG_ROW_H * 2))
    if len(domains) == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(f'Cross-patch {label} — All bias domains\n{desc}',
                 fontsize=FS_SUPTITLE, fontweight='bold')
    for col, domain in enumerate(domains):
        res = domain_results[domain]
        _draw_bars(axes[0, col],
                   res['bias_mean'], res['mlp_mean'], res['attn_mean'],
                   STATES_LABELS, BAR_COLORS, res['num_layer'],
                   'Layer', Y_LABEL_BARS, f'{domain.title()} — states')
        _draw_bars(axes[1, col],
                   res['bias_mean'], res['pre_blank_mean'], res['blank_mean'],
                   WORDS_LABELS, BAR_COLORS, res['num_layer'],
                   'Layer', Y_LABEL_BARS, f'{domain.title()} — words')
        for row in range(2):
            axes[row, col].axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            _low_sig_decorate(axes[row, col], res)
    plt.tight_layout()
    _savepdf(fig, os.path.join(out_dir, 'composite-all.pdf'))


def save_cross_patch_comparison(all_direction_results, domains, out_dir, family_label=''):
    """
    Side-by-side comparison of both cross-patch directions for each domain.
    Y-axis is fixed across directions so the plots are directly comparable.

    all_direction_results : {direction_key: {domain: result_dict}}
    domains               : list of domains to include
    out_dir               : plots/cross_patch/{family}/
    family_label          : display name of the model family, shown in suptitles

    Outputs:
      {domain}-directions-states.pdf   pre→post vs post→pre (states restore)
      {domain}-directions-words.pdf    pre→post vs post→pre (word-token positions)
    """
    direction_keys = list(all_direction_results.keys())

    for plot_type in ('states', 'words'):
        labels_list = STATES_LABELS if plot_type == 'states' else WORDS_LABELS

        for domain in domains:
            # collect one subplot per direction that has data for this domain
            subplots = []
            for dk in direction_keys:
                res = all_direction_results[dk].get(domain)
                if res is None:
                    continue
                dir_label = CROSS_PATCH_CONFIGS[dk]['label']
                if plot_type == 'states':
                    r2, r3 = res['mlp_mean'], res['attn_mean']
                else:
                    r2, r3 = res['pre_blank_mean'], res['blank_mean']
                subplots.append((dir_label, res['bias_mean'], r2, r3,
                                 res['num_layer'], res['low_sig']))

            if not subplots:
                continue

            # shared Y-axis so both directions are directly comparable
            all_vals = np.concatenate([np.concatenate([r1, r2, r3])
                                       for _, r1, r2, r3, _, _ in subplots])
            margin = (np.nanmax(all_vals) - np.nanmin(all_vals)) * 0.12 or 0.05
            y_min, y_max = np.nanmin(all_vals) - margin, np.nanmax(all_vals) + margin

            n = len(subplots)
            fig, axes = plt.subplots(1, n,
                                     figsize=(FIG_GRID_W_PER_COL * n, FIG_ROW_H))
            if n == 1:
                axes = [axes]
            fam = f' [{family_label}]' if family_label else ''
            fig.suptitle(
                f'Cross-patch comparison{fam} — {domain.capitalize()} bias ({plot_type})\n'
                'Y-axis fixed across directions for direct comparison.',
                fontsize=FS_SUPTITLE, fontweight='bold')

            for ax, (dir_label, r1, r2, r3, nl, low_sig) in zip(axes, subplots):
                _draw_bars(ax, r1, r2, r3, labels_list, BAR_COLORS, nl,
                           'Layer', Y_LABEL_BARS, dir_label)
                ax.set_ylim(y_min, y_max)
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
                if low_sig:
                    ax.set_facecolor(LOW_SIG_BG)
                    ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                            fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

            plt.tight_layout()
            _savepdf(fig, os.path.join(out_dir, f'{domain}-directions-{plot_type}.pdf'))


def load_within_model_from_zip(zf, zip_names_all, model_name, org, checkpoint, domain,
                               num_sample=None):
    """
    Load within-model causal tracing results from the shared zip for one
    (model, checkpoint, domain) triple.  Returns a result dict in the same
    format as load_cross_patch_domain, or None on failure.
    """
    prefix    = zip_cases_prefix(org, model_name, checkpoint, domain)
    all_names = [n for n in zip_names_all if n.startswith(prefix) and n.endswith('.npz')]
    if not all_names:
        print(f'    [4panel] No zip data for {prefix}; skipping.')
        return None

    basenames    = [os.path.basename(n) for n in all_names]
    single_b, attn_b, mlp_b = partition_names(basenames)
    single_b, attn_b, mlp_b = subsample_aligned(single_b, attn_b, mlp_b, num_sample)
    name_map     = {os.path.basename(n): n for n in all_names}
    single_items = [name_map[b] for b in single_b if b in name_map]
    attn_items   = [name_map[b] for b in attn_b   if b in name_map]
    mlp_items    = [name_map[b] for b in mlp_b    if b in name_map]
    loader       = lambda p: load_npz_zip(zf, p)

    print(f'    single={len(single_items)}, attn={len(attn_items)}, mlp={len(mlp_items)}')
    if not single_items:
        return None

    expected_model = f'{org}/{model_name}'
    expected = {
        'score_metric': SCORE_METRIC,
        'source_model': expected_model,
        'target_model': expected_model,
        'direction': f'{expected_model} -> {expected_model}',
    }
    if not _validate_result_group(
            (single_items, attn_items, mlp_items), loader,
            expected, '4panel'):
        return None

    try:
        num_layer = loader(single_items[0])['scores'].shape[-1]
    except Exception as ex:
        print(f'    [4panel] Cannot read sample: {ex}')
        return None

    bias_mean, pre_blank_mean, blank_mean, n_cases, mean_high, mean_low = \
        collect_scores(single_items, loader)
    attn_mean, _, _, _, _, _ = collect_scores(attn_items, loader)
    mlp_mean,  _, _, _, _, _ = collect_scores(mlp_items,  loader)

    if bias_mean is None:
        return None

    zero           = np.zeros(num_layer)
    attn_mean      = attn_mean      if attn_mean      is not None else zero
    mlp_mean       = mlp_mean       if mlp_mean       is not None else zero
    pre_blank_mean = pre_blank_mean if pre_blank_mean is not None else zero
    blank_mean     = blank_mean     if blank_mean     is not None else zero

    effect_gap = mean_high - mean_low
    return {
        'bias_mean':      bias_mean,
        'pre_blank_mean': pre_blank_mean,
        'blank_mean':     blank_mean,
        'attn_mean':      attn_mean,
        'mlp_mean':       mlp_mean,
        'n_cases':        n_cases,
        'mean_high':      mean_high,
        'mean_low':       mean_low,
        'effect_gap':     effect_gap,
        'low_sig':        effect_gap < LOW_SIGNAL,
        'num_layer':      num_layer,
    }


def load_within_model_from_local(model_name, org, checkpoint, domain, num_sample=None):
    """
    Load within-model causal tracing results from the local filesystem for one
    (model, checkpoint, domain) triple.  Returns a result dict in the same
    format as load_within_model_from_zip, or None on failure.
    """
    cases_dir = local_cases_dir(model_name, org, checkpoint, domain)
    if not os.path.isdir(cases_dir):
        print(f'    [4panel] No local directory: {cases_dir}; skipping.')
        return None

    all_local    = sorted(os.listdir(cases_dir))
    single_b, attn_b, mlp_b = partition_names(all_local)
    single_b, attn_b, mlp_b = subsample_aligned(single_b, attn_b, mlp_b, num_sample)
    single_items = [os.path.join(cases_dir, b) for b in single_b]
    attn_items   = [os.path.join(cases_dir, b) for b in attn_b]
    mlp_items    = [os.path.join(cases_dir, b) for b in mlp_b]
    loader       = load_npz_local

    print(f'    single={len(single_items)}, attn={len(attn_items)}, mlp={len(mlp_items)}')
    if not single_items:
        return None

    expected_model = f'{org}/{model_name}'
    expected = {
        'score_metric': SCORE_METRIC,
        'source_model': expected_model,
        'target_model': expected_model,
        'direction': f'{expected_model} -> {expected_model}',
    }
    if not _validate_result_group(
            (single_items, attn_items, mlp_items), loader,
            expected, '4panel'):
        return None

    try:
        num_layer = loader(single_items[0])['scores'].shape[-1]
    except Exception as ex:
        print(f'    [4panel] Cannot read sample: {ex}')
        return None

    bias_mean, pre_blank_mean, blank_mean, n_cases, mean_high, mean_low = \
        collect_scores(single_items, loader)
    attn_mean, _, _, _, _, _ = collect_scores(attn_items, loader)
    mlp_mean,  _, _, _, _, _ = collect_scores(mlp_items,  loader)

    if bias_mean is None:
        return None

    zero           = np.zeros(num_layer)
    attn_mean      = attn_mean      if attn_mean      is not None else zero
    mlp_mean       = mlp_mean       if mlp_mean       is not None else zero
    pre_blank_mean = pre_blank_mean if pre_blank_mean is not None else zero
    blank_mean     = blank_mean     if blank_mean     is not None else zero

    effect_gap = mean_high - mean_low
    return {
        'bias_mean':      bias_mean,
        'pre_blank_mean': pre_blank_mean,
        'blank_mean':     blank_mean,
        'attn_mean':      attn_mean,
        'mlp_mean':       mlp_mean,
        'n_cases':        n_cases,
        'mean_high':      mean_high,
        'mean_low':       mean_low,
        'effect_gap':     effect_gap,
        'low_sig':        effect_gap < LOW_SIGNAL,
        'num_layer':      num_layer,
    }


def save_cross_patch_4panel(within_model_panels, all_direction_results, domains, out_dir,
                            inst_label='OLMo Instruct\n(step2600)', file_prefix='4panel',
                            extra_out_dir=None):
    """
    4-panel comparison per domain (and composite over all domains):

      Panel 1 — OLMo Stage 2 last checkpoint  (within-model causal tracing)
      Panel 2 — OLMo Instruct checkpoint       (within-model causal tracing)
      Panel 3 — Pre → Post cross-patch
      Panel 4 — Post → Pre cross-patch

    Y-axis is fixed across all 4 panels so they are directly comparable.

    within_model_panels  : {domain: {'base_last': result_dict, 'inst_last': result_dict}}
    all_direction_results: {direction_key: {domain: result_dict}}
    domains              : ordered list of domains to include
    out_dir              : plots/cross_patch/
    inst_label           : panel title for the instruct column
    file_prefix          : prefix for output filenames (default '4panel')

    Output files:
      {file_prefix}-{domain}-states.pdf
      {file_prefix}-{domain}-words.pdf
      {file_prefix}-composite-states.pdf
      {file_prefix}-composite-words.pdf
    """
    PANEL_DEFS = [
        ('base_last',    'OLMo-2-0425-1B\n(pre)',  'within'),
        ('inst_last',  inst_label,               'within'),
        ('pre_to_post', 'Pre → Post',             'cross'),
        ('post_to_pre', 'Post → Pre',             'cross'),
    ]

    def _get_res(key, src, domain):
        if src == 'within':
            return within_model_panels.get(domain, {}).get(key)
        return all_direction_results.get(key, {}).get(domain)

    def _shared_ylim(panels_data, plot_type):
        all_vals = []
        for res in panels_data:
            if res is None:
                continue
            if plot_type == 'states':
                all_vals.extend([res['bias_mean'], res['mlp_mean'], res['attn_mean']])
            else:
                all_vals.extend([res['bias_mean'], res['pre_blank_mean'], res['blank_mean']])
        if not all_vals:
            return None, None
        flat   = np.concatenate(all_vals)
        margin = (np.nanmax(flat) - np.nanmin(flat)) * 0.12 or 0.05
        return np.nanmin(flat) - margin, np.nanmax(flat) + margin

    def _fill_ax(ax, res, plot_type, labels_list, title, y_min, y_max, show_ylabel=True):
        if res is None:
            ax.set_visible(False)
            return
        if plot_type == 'states':
            r1, r2, r3 = res['bias_mean'], res['mlp_mean'], res['attn_mean']
        else:
            r1, r2, r3 = res['bias_mean'], res['pre_blank_mean'], res['blank_mean']
        ylabel = Y_LABEL_BARS if show_ylabel else ''
        _draw_bars(ax, r1, r2, r3, labels_list, BAR_COLORS, res['num_layer'],
                   'Layer', ylabel, title,
                   fs_label=FS_LABEL+6, fs_tick=FS_TICK+6, fs_title=FS_TITLE+5)
        ax.tick_params(axis='y', labelsize=FS_TICK+6)
        ax.set_xlabel('Layer', fontsize=FS_LABEL+5, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
        if res['low_sig']:
            ax.set_facecolor(LOW_SIG_BG)
            ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                    fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

    for plot_type in ('states', 'words'):
        labels_list = STATES_LABELS if plot_type == 'states' else WORDS_LABELS

        def _shared_legend(fig):
            legend_handles = [Patch(facecolor=c, edgecolor='gray', label=l)
                              for c, l in zip(BAR_COLORS, labels_list)]
            fig.legend(handles=legend_handles, loc='lower center',
                       bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=FS_LEGEND + 7, frameon=True)

        def _label_ax(ax, letter):
            ax.text(0.02, 0.98, letter, transform=ax.transAxes,
                    fontsize=FS_TITLE + 2, fontweight='bold', va='top', ha='left',
                    zorder=10, clip_on=False,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              alpha=0.85, edgecolor='gray', linewidth=0.5))

        def _draw_4panel(panels_data, suptitle, out_path, extra_path=None):
            y_min, y_max = _shared_ylim(panels_data, plot_type)
            if y_min is None:
                return
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for i, (ax, (key, label, src), res) in enumerate(
                    zip(axes, PANEL_DEFS, panels_data)):
                _fill_ax(ax, res, plot_type, labels_list,
                         f'({"abcd"[i]}) {label}', y_min, y_max,
                         show_ylabel=(i == 0))
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            _shared_legend(fig)
            fig.tight_layout(rect=[0, 0.10, 1, 1.0])
            if extra_path:
                os.makedirs(os.path.dirname(extra_path), exist_ok=True)
                fig.savefig(extra_path, format='pdf', bbox_inches='tight')
                print(f'    Saved: {extra_path}')
            _savepdf(fig, out_path)

        def _draw_2panel(defs_subset, data_subset, suptitle, out_path):
            # Matches the 4-panel styling exactly (letter-prefixed titles, ylabel
            # only on the left panel, shared bottom legend, no suptitle); each
            # 2-panel figure carries its own legend.
            y_min, y_max = _shared_ylim(data_subset, plot_type)
            if y_min is None:
                return
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for i, (ax, (key, label, src), res) in enumerate(
                    zip(axes, defs_subset, data_subset)):
                _fill_ax(ax, res, plot_type, labels_list,
                         f'({"ab"[i]}) {label}', y_min, y_max,
                         show_ylabel=(i == 0))
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            _shared_legend(fig)
            fig.tight_layout(rect=[0, 0.10, 1, 1.0])
            _savepdf(fig, out_path)

        # ── per-domain figures ────────────────────────────────────────────────
        for domain in domains:
            panels_data = [_get_res(key, src, domain) for key, _, src in PANEL_DEFS]
            dom = domain.capitalize()

            # Layout 1: 2×2 four-panel (A=Stage2, B=Instruct, C=Pre→Post, D=Post→Pre)
            _fname = f'{file_prefix}-{domain}-{plot_type}.pdf'
            _draw_4panel(
                panels_data,
                f'{dom} bias — 4-panel cross-patch comparison ({plot_type})\n'
                'Cols 1–2: within-model  |  Cols 3–4: cross-model  |  Y-axis fixed across panels',
                os.path.join(out_dir, _fname),
                extra_path=os.path.join(extra_out_dir, _fname) if extra_out_dir else None)

            # Layout 2: within-model — Base (main) vs Instruct (step2000)
            _draw_2panel(
                PANEL_DEFS[:2], panels_data[:2],
                f'{dom} bias — within-model: Stage 2 vs Instruct ({plot_type})\n'
                'Y-axis fixed across panels',
                os.path.join(out_dir, f'{file_prefix}-{domain}-{plot_type}-within.pdf'))

            # Layout 3: cross-patch — Pre→Post vs Post→Pre
            _draw_2panel(
                PANEL_DEFS[2:], panels_data[2:],
                f'{dom} bias — cross-patch: Pre→Post vs Post→Pre ({plot_type})\n'
                'Y-axis fixed across panels',
                os.path.join(out_dir, f'{file_prefix}-{domain}-{plot_type}-cross.pdf'))

        # ── composite: all domains, rows=domains cols=panels ─────────────────
        n_domains = len(domains)
        fig, axes = plt.subplots(n_domains, 4,
                                 figsize=(FIG_BAR_W_PER_COL * 4, FIG_ROW_H * n_domains))
        if n_domains == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(
            f'Cross-patch 4-panel comparison — all domains ({plot_type})\n'
            'Columns: Stage2-last | Instruct-last | Pre→Post | Post→Pre  '
            '|  Y-axis fixed per row',
            fontsize=FS_SUPTITLE, fontweight='bold')
        for row, domain in enumerate(domains):
            panels_data = [_get_res(key, src, domain) for key, _, src in PANEL_DEFS]
            y_min, y_max = _shared_ylim(panels_data, plot_type)
            if y_min is None:
                for col in range(4):
                    axes[row, col].set_visible(False)
                continue
            for col, ((key, label, src), res) in enumerate(zip(PANEL_DEFS, panels_data)):
                row_label = f'{domain.capitalize()} — {label}'
                _fill_ax(axes[row, col], res, plot_type, labels_list, row_label, y_min, y_max)
        plt.tight_layout()
        _savepdf(fig, os.path.join(out_dir, f'{file_prefix}-composite-{plot_type}.pdf'))


# ── appendix data helper ──────────────────────────────────────────────────────

def _load_from_main_zip(main_zf, domain, num_sample=None):
    """
    Load causal-tracing result dict from main.zip for one domain.
    Path structure in zip: main/{domain}/causal_trace/cases/
    Returns a result dict in the same format as load_cross_patch_domain, or None.
    """
    prefix = f'main/{domain}/causal_trace/cases/'
    names  = [n for n in main_zf.namelist() if n.startswith(prefix) and n.endswith('.npz')]
    if not names:
        print(f'    [main.zip] No files for {domain} at {prefix}')
        return None
    basenames    = [os.path.basename(n) for n in names]
    single_b, attn_b, mlp_b = partition_names(basenames)
    single_b, attn_b, mlp_b = subsample_aligned(single_b, attn_b, mlp_b, num_sample)
    name_map     = {os.path.basename(n): n for n in names}
    single_items = [name_map[b] for b in single_b if b in name_map]
    attn_items   = [name_map[b] for b in attn_b   if b in name_map]
    mlp_items    = [name_map[b] for b in mlp_b    if b in name_map]
    loader       = lambda p: load_npz_zip(main_zf, p)
    print(f'    [main.zip] {domain}: single={len(single_items)}, attn={len(attn_items)}, mlp={len(mlp_items)}')
    if not single_items:
        return None
    expected_model = 'allenai/OLMo-2-0425-1B'
    expected = {
        'score_metric': SCORE_METRIC,
        'source_model': expected_model,
        'target_model': expected_model,
        'direction': f'{expected_model} -> {expected_model}',
    }
    if not _validate_result_group(
            (single_items, attn_items, mlp_items), loader,
            expected, 'main.zip'):
        return None
    try:
        num_layer = loader(single_items[0])['scores'].shape[-1]
    except Exception as ex:
        print(f'    [main.zip] Cannot read sample: {ex}')
        return None
    bias_mean, pre_blank_mean, blank_mean, n_cases, mean_high, mean_low = \
        collect_scores(single_items, loader)
    attn_mean, _, _, _, _, _ = collect_scores(attn_items, loader)
    mlp_mean,  _, _, _, _, _ = collect_scores(mlp_items,  loader)
    if bias_mean is None:
        return None
    zero           = np.zeros(num_layer)
    attn_mean      = attn_mean      if attn_mean      is not None else zero
    mlp_mean       = mlp_mean       if mlp_mean       is not None else zero
    pre_blank_mean = pre_blank_mean if pre_blank_mean is not None else zero
    blank_mean     = blank_mean     if blank_mean     is not None else zero
    effect_gap = mean_high - mean_low
    return {
        'bias_mean':      bias_mean,
        'pre_blank_mean': pre_blank_mean,
        'blank_mean':     blank_mean,
        'attn_mean':      attn_mean,
        'mlp_mean':       mlp_mean,
        'n_cases':        n_cases,
        'mean_high':      mean_high,
        'mean_low':       mean_low,
        'effect_gap':     effect_gap,
        'low_sig':        effect_gap < LOW_SIGNAL,
        'num_layer':      num_layer,
    }


# ── appendix grid helpers ─────────────────────────────────────────────────────

def _row_ylim(results_dict, paper_domains, *keys):
    """Per-row (y_min, y_max) pooled across all domains for the given key set."""
    all_vals = []
    for d in paper_domains:
        res = results_dict.get(d)
        if res is None:
            continue
        for k in keys:
            v = res.get(k)
            if v is not None:
                all_vals.append(v)
    if not all_vals:
        return None, None
    flat   = np.concatenate(all_vals)
    margin = (float(flat.max()) - float(flat.min())) * 0.12 or 0.05
    return float(flat.min()) - margin, float(flat.max()) + margin


def _bars_grid(axes, row_specs, paper_domains, row_ylims):
    """
    Fill a rows×cols axes grid with labelled bar charts.

    axes         : 2-D numpy array of Axes, shape (n_rows, len(paper_domains))
    row_specs    : list of (results_dict, plot_type, bar_colors, model_label, type_label)
      plot_type  : 'states' or 'words'
      bar_colors : 3-element list of colors for the three bars
    paper_domains : ordered list of domain strings
    row_ylims    : list of (y_min, y_max) per row; None entries mean auto-scale
    """
    alphabet   = 'abcdefghijklmnopqrstuvwxyz'
    letter_idx = 0

    for row, (results_dict, plot_type, bar_colors, model_label, type_label) in enumerate(row_specs):
        y_min_r, y_max_r = row_ylims[row] if row_ylims is not None else (None, None)
        for col, domain in enumerate(paper_domains):
            ax     = axes[row, col]
            res    = results_dict.get(domain)
            letter = alphabet[letter_idx]
            letter_idx += 1
            title  = f'({letter}) {domain.capitalize()} bias {type_label}\n({model_label})'

            if res is None:
                ax.set_visible(False)
                continue

            if plot_type == 'states':
                r1, r2, r3  = res['bias_mean'], res['mlp_mean'], res['attn_mean']
                labels_list = STATES_LABELS
            else:
                r1, r2, r3  = res['bias_mean'], res['pre_blank_mean'], res['blank_mean']
                labels_list = WORDS_LABELS

            _draw_bars(ax, r1, r2, r3, labels_list, bar_colors, res['num_layer'],
                       'Layer', Y_LABEL_BARS if col == 0 else '', title,
                       fs_label=FS_LABEL+3, fs_tick=FS_TICK+3)

            if y_min_r is not None:
                ax.set_ylim(y_min_r, y_max_r)

            ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4, zorder=0)
            leg = ax.get_legend()
            if leg:
                leg.remove()

            if res['low_sig']:
                ax.set_facecolor(LOW_SIG_BG)
                ax.text(0.98, 0.97, '⚠', transform=ax.transAxes,
                        fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)


def _states_words_legend(fig, bottom_frac=0.10):
    """
    Single centered legend with 2 rows: States (row 1) + Words (row 2), ncol=3.
    States bars use STATES_COLORS; words bars use WORDS_COLORS.
    """
    patches = (
        [Patch(facecolor=STATES_COLORS[i], edgecolor='gray', label=STATES_LABELS[i])
         for i in range(3)] +
        [Patch(facecolor=WORDS_COLORS[i],  edgecolor='gray', label=WORDS_LABELS[i])
         for i in range(3)]
    )
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=3, fontsize=FS_LEGEND + 5, frameon=True)


# ── appendix A1: OLMo base + instruct bars (4×3) ─────────────────────────────

def save_appendix_A1_olmo_bars(out_dir, num_sample=None, main_zf=None):
    """
    Appendix A1: 4×3 grid for OLMo-2-0425-1B (main) and OLMo-2-0425-1B-Instruct (step_2000).
    Row 0 base states · Row 1 base words · Row 2 instruct states · Row 3 instruct words.
    Columns: gender, race, profession.
    States bars use STATES_COLORS; words bars use WORDS_COLORS.
    Single global Y-axis across all 12 panels.
    """
    print('  [A1] Loading OLMo base from main.zip...')
    _own_zf = main_zf is None
    if _own_zf:
        main_zf = zipfile.ZipFile(MAIN_ZIP, 'r')
    base_results = {d: _load_from_main_zip(main_zf, d, num_sample) for d in PAPER_DOMAINS}
    if _own_zf:
        main_zf.close()

    print('  [A1] Loading OLMo instruct from local (step_2000)...')
    inst_results = {d: load_within_model_from_local(
        'OLMo-2-0425-1B-Instruct', 'allenai', 'step_2000', d, num_sample)
        for d in PAPER_DOMAINS}

    row_specs = [
        (base_results, 'states', STATES_COLORS, 'OLMo-2-0425-1B',          'effect of states'),
        (base_results, 'words',  WORDS_COLORS,  'OLMo-2-0425-1B',          'effect of different words'),
        (inst_results, 'states', STATES_COLORS, 'OLMo-2-0425-1B-Instruct', 'effect of states'),
        (inst_results, 'words',  WORDS_COLORS,  'OLMo-2-0425-1B-Instruct', 'effect of different words'),
    ]
    row_ylims = [
        _row_ylim(base_results, PAPER_DOMAINS, 'bias_mean', 'mlp_mean', 'attn_mean'),
        _row_ylim(base_results, PAPER_DOMAINS, 'bias_mean', 'pre_blank_mean', 'blank_mean'),
        _row_ylim(inst_results, PAPER_DOMAINS, 'bias_mean', 'mlp_mean', 'attn_mean'),
        _row_ylim(inst_results, PAPER_DOMAINS, 'bias_mean', 'pre_blank_mean', 'blank_mean'),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(FIG_BAR_W_PER_COL * 3, FIG_ROW_H * 4))
    _bars_grid(axes, row_specs, PAPER_DOMAINS, row_ylims)
    _states_words_legend(fig)
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    _savepdf(fig, os.path.join(out_dir, 'A1-olmo-bars.pdf'))


# ── appendix A2: Pythia bars (2×3) ───────────────────────────────────────────

def save_appendix_A2_pythia_bars(out_dir, num_sample=None):
    """
    Appendix A2: 2×3 grid for Pythia-1B (step143000).
    Row 0 states · Row 1 words. Columns: gender, race, profession.
    States bars use STATES_COLORS; words bars use WORDS_COLORS.
    Single global Y-axis across all 6 panels.
    """
    print('  [A2] Loading Pythia-1B (step143000) from results.zip...')
    pythia_results = {d: load_within_model_from_zip(
        zf, zip_names_all, 'pythia-1b', 'EleutherAI', 'step143000', d, num_sample)
        for d in PAPER_DOMAINS}

    if not any(v is not None for v in pythia_results.values()):
        print('  [A2] No data; skipping.')
        return

    row_specs = [
        (pythia_results, 'states', STATES_COLORS, 'pythia-1b', 'effect of states'),
        (pythia_results, 'words',  WORDS_COLORS,  'pythia-1b', 'effect of different words'),
    ]
    row_ylims = [
        _row_ylim(pythia_results, PAPER_DOMAINS, 'bias_mean', 'mlp_mean', 'attn_mean'),
        _row_ylim(pythia_results, PAPER_DOMAINS, 'bias_mean', 'pre_blank_mean', 'blank_mean'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(FIG_BAR_W_PER_COL * 3, FIG_ROW_H * 2))
    _bars_grid(axes, row_specs, PAPER_DOMAINS, row_ylims)
    _states_words_legend(fig)
    fig.tight_layout(rect=[0, 0.11, 1, 1.0])
    _savepdf(fig, os.path.join(out_dir, 'A2-pythia-bars.pdf'))


# ── appendix A3: NIE line plots (4×3) ────────────────────────────────────────

def save_appendix_A3_nie_lines(out_dir, num_sample=None, main_zf=None):
    """
    Appendix A3: 4×3 NIE layer-profile figure.
    Row 0: OLMo base (solid) vs OLMo Instruct (dashed) — states NIE — 3 domains.
    Row 1: OLMo base (solid) vs OLMo Instruct (dashed) — words  NIE — 3 domains.
    Row 2: OLMo base (solid) vs Pythia (dashed)         — states NIE — 3 domains.
    Row 3: OLMo base (solid) vs Pythia (dashed)         — words  NIE — 3 domains.

    States rows use BAR_COLORS; words rows use WORDS_COLORS.
    States rows share one Y-range; words rows share a separate Y-range.
    """
    from matplotlib.lines import Line2D

    A3_FS_TITLE  = FS_TITLE + 1      # 11
    A3_FS_LABEL  = FS_LABEL  + 5   # 14
    A3_FS_TICK   = FS_TICK   + 4   # 12
    A3_FS_LEGEND = FS_LEGEND + 6   # 13

    print('  [A3] Loading OLMo base from main.zip...')
    _own_zf = main_zf is None
    if _own_zf:
        main_zf = zipfile.ZipFile(MAIN_ZIP, 'r')
    base_results = {d: _load_from_main_zip(main_zf, d, num_sample) for d in PAPER_DOMAINS}
    if _own_zf:
        main_zf.close()

    print('  [A3] Loading OLMo instruct from local (step_2000)...')
    inst_results = {d: load_within_model_from_local(
        'OLMo-2-0425-1B-Instruct', 'allenai', 'step_2000', d, num_sample)
        for d in PAPER_DOMAINS}

    print('  [A3] Loading Pythia (step143000) from results.zip...')
    pythia_results = {d: load_within_model_from_zip(
        zf, zip_names_all, 'pythia-1b', 'EleutherAI', 'step143000', d, num_sample)
        for d in PAPER_DOMAINS}

    STATES_KEYS   = ('bias', 'mlp', 'attn')
    STATES_LABELS_SHORT = ['States', 'MLP window', 'Attn window']
    WORDS_KEYS    = ('bias', 'pre_blank', 'blank')
    WORDS_LABELS_SHORT  = [
        'Effect of subject token',
        'Effect of pre-target token',
        'Effect of target token',
    ]

    def _nie(res, key):
        """Compute NIE for a given data key in a result dict."""
        key_map = {
            'bias':      'bias_mean',
            'mlp':       'mlp_mean',
            'attn':      'attn_mean',
            'pre_blank': 'pre_blank_mean',
            'blank':     'blank_mean',
        }
        if res is None:
            return None
        arr = res.get(key_map[key])
        if arr is None:
            return None
        return normalized_indirect_effect(arr, res['mean_low'], res['effect_gap'])   # None if gap <= 0

    # compute Y ranges separately for states rows and words rows
    states_vals, words_vals = [], []
    for results_dict in (base_results, inst_results, pythia_results):
        for d in PAPER_DOMAINS:
            res = results_dict.get(d)
            for k in STATES_KEYS:
                v = _nie(res, k)
                if v is not None:
                    states_vals.extend(v.tolist())
            for k in WORDS_KEYS:
                v = _nie(res, k)
                if v is not None:
                    words_vals.extend(v.tolist())

    if not states_vals and not words_vals:
        print('  [A3] No valid NIE data; skipping.')
        return

    def _yrange(vals):
        if not vals:
            return -0.1, 1.1
        m = (max(vals) - min(vals)) * 0.12 or 0.05
        return min(vals) - m, max(vals) + m

    states_ymin, states_ymax = _yrange(states_vals)
    words_ymin,  words_ymax  = _yrange(words_vals)

    # 4 rows: (plot_type, model1_results, label1, model2_results, label2)
    ROW_DEFS = [
        ('states', base_results, 'OLMo-2-0425-1B (solid)',
                   inst_results,    'OLMo-2-0425-1B-Instruct (dashed)'),
        ('words',  base_results, 'OLMo-2-0425-1B (solid)',
                   inst_results,    'OLMo-2-0425-1B-Instruct (dashed)'),
        ('states', base_results, 'OLMo-2-0425-1B (solid)',
                   pythia_results,  'Pythia-1B (dashed)'),
        ('words',  base_results, 'OLMo-2-0425-1B (solid)',
                   pythia_results,  'Pythia-1B (dashed)'),
    ]
    letters_all = list('abcdefghijkl')

    fig, axes = plt.subplots(4, 3, figsize=(FIG_LINE_W_PER_PAN * 3, (FIG_ROW_H + 0.5) * 4))
    letter_idx = 0

    for row, (plot_type, model1_results, model1_label, model2_results, model2_label) in enumerate(ROW_DEFS):
        is_states = (plot_type == 'states')
        keys      = STATES_KEYS   if is_states else WORDS_KEYS
        labels    = STATES_LABELS_SHORT if is_states else WORDS_LABELS_SHORT
        colors    = BAR_COLORS    if is_states else WORDS_COLORS
        ymin      = states_ymin   if is_states else words_ymin
        ymax      = states_ymax   if is_states else words_ymax
        type_tag  = 'States NIE'  if is_states else 'Words NIE'

        for col, domain in enumerate(PAPER_DOMAINS):
            ax     = axes[row, col]
            letter = letters_all[letter_idx]
            letter_idx += 1
            res1 = model1_results.get(domain)
            res2 = model2_results.get(domain)

            for ki, (key, cond_label) in enumerate(zip(keys, labels)):
                color = colors[ki]
                nie1  = _nie(res1, key)
                nie2  = _nie(res2, key)
                if nie1 is not None:
                    xs = np.arange(len(nie1))
                    ax.plot(xs, nie1, color=color, linewidth=2.0, linestyle='-',
                            marker='o', markersize=3)
                if nie2 is not None:
                    xs = np.arange(len(nie2))
                    ax.plot(xs, nie2, color=color, linewidth=2.0, linestyle='--',
                            marker='s', markersize=3)

            ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
            ax.set_title(
                f'({letter}) {domain.capitalize()} — {type_tag}\n'
                f'{model1_label} vs {model2_label}',
                fontsize=A3_FS_TITLE)
            ax.set_xlabel('Layer', fontsize=A3_FS_LABEL)
            if col == 0:
                ax.set_ylabel(Y_LABEL_NIE, fontsize=A3_FS_LABEL)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(np.arange(0, 16, max(1, 16 // 8)))
            ax.tick_params(labelsize=A3_FS_TICK)
            ax.grid(alpha=0.2)

    # two legend blocks at bottom: states (BAR_COLORS) and words (WORDS_COLORS)
    legend_handles = []
    for ki, cond_label in enumerate(STATES_LABELS_SHORT):
        c = BAR_COLORS[ki]
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='-',
                   marker='o', markersize=4, label=f'{cond_label} — solid'))
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='--',
                   marker='s', markersize=4, label=f'{cond_label} — dashed'))
    # separator
    legend_handles.append(Line2D([0], [0], color='none', label=''))
    for ki, cond_label in enumerate(WORDS_LABELS_SHORT):
        c = WORDS_COLORS[ki]
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='-',
                   marker='o', markersize=4, label=f'{cond_label} — solid'))
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='--',
                   marker='s', markersize=4, label=f'{cond_label} — dashed'))

    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=A3_FS_LEGEND, frameon=True,
               labelspacing=1.0, handlelength=3.0, handletextpad=0.8, columnspacing=2.5)
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    _savepdf(fig, os.path.join(out_dir, 'A3-nie-lines.pdf'))


# ── main body: NIE overlay (gender, single panel) ────────────────────────────

def save_main_body_nie_overlay(out_dir, num_sample=None, main_zf=None):
    """
    Main body figure: single NIE panel for gender domain.
    OLMo-2-0425-1B (solid) vs OLMo-2-0425-1B-Instruct (dashed).
    3 restore conditions × 2 models = 6 lines.
    Saved to out_dir/nie-overlay-gender.pdf
    """
    from matplotlib.lines import Line2D

    print('  [NIE overlay] Loading OLMo base from main.zip...')
    _own_zf = main_zf is None
    if _own_zf:
        main_zf = zipfile.ZipFile(MAIN_ZIP, 'r')
    base_res = _load_from_main_zip(main_zf, 'gender', num_sample)
    if _own_zf:
        main_zf.close()

    print('  [NIE overlay] Loading OLMo instruct (step_2000)...')
    inst_res = load_within_model_from_local(
        'OLMo-2-0425-1B-Instruct', 'allenai', 'step_2000', 'gender', num_sample)

    SCORE_KEYS = ('states_score', 'mlp_score', 'attn_score')
    NIE_LABELS = ['States', 'MLP window', 'Attn window']

    def _nie(res, key):
        key_map = {'states_score': 'bias_mean', 'mlp_score': 'mlp_mean', 'attn_score': 'attn_mean'}
        if res is None:
            return None
        return normalized_indirect_effect(res[key_map[key]], res['mean_low'], res['effect_gap'])   # None if gap <= 0

    all_vals = []
    for key in SCORE_KEYS:
        for res in (base_res, inst_res):
            v = _nie(res, key)
            if v is not None:
                all_vals.extend(v.tolist())

    if not all_vals:
        print('  [NIE overlay] No data; skipping.')
        return

    margin = (max(all_vals) - min(all_vals)) * 0.12 or 0.05
    y_min  = min(all_vals) - margin
    y_max  = max(all_vals) + margin

    fig, ax = plt.subplots(1, 1, figsize=(7.0, FIG_ROW_H + 0.5))

    for ki, (key, cond_label) in enumerate(zip(SCORE_KEYS, NIE_LABELS)):
        color = BAR_COLORS[ki]
        nie_b = _nie(base_res, key)
        nie_i = _nie(inst_res, key)
        xs    = np.arange(16)
        if nie_b is not None:
            ax.plot(xs, nie_b, color=color, linewidth=2.0, linestyle='-',
                    marker='o', markersize=3)
        if nie_i is not None:
            ax.plot(xs, nie_i, color=color, linewidth=2.0, linestyle='--',
                    marker='s', markersize=3)

    ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
    ax.set_title('(a) Gender\nOLMo-2-0425-1B (solid) vs OLMo-2-0425-1B-Instruct (dashed)',
                 fontsize=FS_TITLE+3)
    ax.set_xlabel('Layer', fontsize=FS_LABEL+5)
    ax.set_ylabel('NIE (normalized indirect effect)', fontsize=FS_LABEL+5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0, 16, 2))
    ax.tick_params(labelsize=FS_TICK+4)
    ax.grid(alpha=0.2)

    legend_handles = []
    for ki, cond_label in enumerate(NIE_LABELS):
        c = BAR_COLORS[ki]
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='-',
                   marker='o', markersize=4, label=f'{cond_label} — Base (solid)'))
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='--',
                   marker='s', markersize=4, label=f'{cond_label} — Instruct (dashed)'))
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=FS_LEGEND + 5, frameon=True,
               labelspacing=0.4, handlelength=1.8, handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout(rect=[0, 0.17, 1, 1.0])
    _savepdf(fig, os.path.join(out_dir, 'nie-overlay-gender.pdf'))


def save_main_body_pre_post_crosspatch(out_dir, num_sample=None, main_zf=None):
    """
    Main-body 2-panel gender figure (1×2, shared y-axis, no suptitle).

      (a) Reference — within-model NIE overlay, 3 restore conditions:
          OLMo-2-0425-1B (Pre, solid) vs OLMo-2-0425-1B-Instruct (Post, dashed).
      (b) States NIE only, four solid lines in distinct colors:
          Pre      = within-model base   (main)
          Post     = within-model instruct (step_2000)
          Pre→Post = cross-patch, Pre activations → Post model
          Post→Pre = cross-patch, Post activations → Pre model

    Every curve is NIE = (score − low) / (high − low), normalized to its own
    recipient's clean−corrupted gap (the cross-patch runs store the recipient's
    baselines), so all lines are directly comparable. Font sizes match
    nie-overlay-gender.pdf. Saved to out_dir/pre-post-crosspatch-gender.pdf
    """
    from matplotlib.lines import Line2D

    FS_T = FS_TITLE + 3   # 13
    FS_L = FS_LABEL + 5   # 14
    FS_K = FS_TICK  + 4   # 12
    FS_G = FS_LEGEND + 5  # 12

    DOMAIN = 'gender'

    print('  [pre-post-cp] Loading Pre (base, main.zip)...')
    _own_zf = main_zf is None
    if _own_zf:
        main_zf = zipfile.ZipFile(MAIN_ZIP, 'r')
    pre_res = _load_from_main_zip(main_zf, DOMAIN, num_sample)
    if _own_zf:
        main_zf.close()

    print('  [pre-post-cp] Loading Post (instruct step_2000) + cross-patch...')
    post_res   = load_within_model_from_local(
        'OLMo-2-0425-1B-Instruct', 'allenai', 'step_2000', DOMAIN, num_sample)
    p2post_res = load_cross_patch_domain('pre_to_post', DOMAIN, num_sample)
    p2pre_res  = load_cross_patch_domain('post_to_pre', DOMAIN, num_sample)

    def _nie(res, key):
        key_map = {'bias': 'bias_mean', 'mlp': 'mlp_mean', 'attn': 'attn_mean'}
        if res is None:
            return None
        return normalized_indirect_effect(res[key_map[key]], res['mean_low'], res['effect_gap'])   # None if gap <= 0

    # (a) within-model overlay: 3 conditions, Pre solid / Post dashed
    COND_KEYS   = ('bias', 'mlp', 'attn')
    COND_LABELS = ['States', 'MLP window', 'Attn window']

    # (b) States-only, four distinct colors
    B_SERIES = [
        ('Pre',      pre_res,    BASE_COLOR,     'o'),
        ('Post',     post_res,   INSTRUCT_COLOR, 's'),
        ('Pre→Post', p2post_res, '#2E7D32',      '^'),
        ('Post→Pre', p2pre_res,  '#6A1B9A',      'D'),
    ]

    # shared y-range across both panels
    all_vals = []
    for key in COND_KEYS:
        for res in (pre_res, post_res):
            v = _nie(res, key)
            if v is not None:
                all_vals.extend(v.tolist())
    for _, res, _, _ in B_SERIES:
        v = _nie(res, 'bias')
        if v is not None:
            all_vals.extend(v.tolist())
    if not all_vals:
        print('  [pre-post-cp] No data; skipping.')
        return
    margin = (max(all_vals) - min(all_vals)) * 0.12 or 0.05
    y_min, y_max = min(all_vals) - margin, max(all_vals) + margin

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, FIG_ROW_H + 1.0), sharey=True)

    # ── panel (a) ──────────────────────────────────────────────────────────────
    for ki, key in enumerate(COND_KEYS):
        color = BAR_COLORS[ki]
        nie_pre  = _nie(pre_res, key)
        nie_post = _nie(post_res, key)
        if nie_pre is not None:
            ax_a.plot(np.arange(len(nie_pre)), nie_pre, color=color, linewidth=2.0,
                      linestyle='-', marker='o', markersize=3)
        if nie_post is not None:
            ax_a.plot(np.arange(len(nie_post)), nie_post, color=color, linewidth=2.0,
                      linestyle='--', marker='s', markersize=3)
    ax_a.axhline(0, color='black', linewidth=0.7, alpha=0.4)
    ax_a.set_title('(a) Gender — OLMo-2-0425-1B (Pre) solid vs\n'
                   'OLMo-2-0425-1B-Instruct (Post) dashed', fontsize=FS_T)
    ax_a.set_xlabel('Layer', fontsize=FS_L)
    ax_a.set_xticks(np.arange(0, 16, 2))
    ax_a.tick_params(labelsize=FS_K)
    ax_a.grid(alpha=0.2)
    handles_a = []
    for ki, cond in enumerate(COND_LABELS):
        c = BAR_COLORS[ki]
        handles_a.append(Line2D([0], [0], color=c, linewidth=2, linestyle='-',
                                marker='o', markersize=4, label=f'{cond} — Pre (solid)'))
        handles_a.append(Line2D([0], [0], color=c, linewidth=2, linestyle='--',
                                marker='s', markersize=4, label=f'{cond} — Post (dashed)'))
    ax_a.legend(handles=handles_a, fontsize=FS_G, frameon=True, ncol=2,
                loc='upper center', bbox_to_anchor=(0.5, -0.22),
                labelspacing=0.3, handlelength=2.0, columnspacing=1.2)

    # ── panel (b) ──────────────────────────────────────────────────────────────
    handles_b = []
    for name, res, color, mk in B_SERIES:
        nie = _nie(res, 'bias')
        if nie is not None:
            ax_b.plot(np.arange(len(nie)), nie, color=color, linewidth=2.0,
                      linestyle='-', marker=mk, markersize=4)
        handles_b.append(Line2D([0], [0], color=color, linewidth=2, linestyle='-',
                                marker=mk, markersize=5, label=f'States — {name}'))
    ax_b.axhline(0, color='black', linewidth=0.7, alpha=0.4)
    ax_b.set_title('(b) Gender — Pre, Post, Pre→Post, Post→Pre\n(States NIE)',
                   fontsize=FS_T)
    ax_b.set_xlabel('Layer', fontsize=FS_L)
    ax_b.set_xticks(np.arange(0, 16, 2))
    ax_b.tick_params(labelsize=FS_K, labelleft=True)   # keep y-ticks despite sharey
    ax_b.grid(alpha=0.2)
    ax_b.legend(handles=handles_b, fontsize=FS_G, frameon=True, ncol=2,
                loc='upper center', bbox_to_anchor=(0.5, -0.22),
                labelspacing=0.3, handlelength=2.0, columnspacing=1.2)

    ax_a.set_ylim(y_min, y_max)
    fig.supylabel(Y_LABEL_NIE, fontsize=FS_L)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)   # tighten gap between the two panels
    _savepdf(fig, os.path.join(out_dir, 'pre-post-crosspatch-gender.pdf'))


# ── appendix A7: cross-patch NIE overlay (within-model vs cross-patch) ────────

def save_crosspatch_nie_overlay(out_dir, num_sample=None, main_zf=None):
    """
    4×3 NIE layer-profile overlay for cross-model patching.
    Solid = within-model recipient; dashed = cross-patch into that recipient.
    Both curves are normalized by the recipient's own clean−corrupted gap
    (the cross-patch run stores the recipient's high/low baselines — the donor
    only sets the restoration value), so they are directly comparable.

      Row 0: Pre→Post — Post within-model (solid) vs Pre→Post cross-patch (dashed) — states
      Row 1: Pre→Post — same pair — words
      Row 2: Post→Pre — Pre  within-model (solid) vs Post→Pre cross-patch (dashed) — states
      Row 3: Post→Pre — same pair — words

    Columns = gender, race, profession. States rows share one Y-range; words
    rows share another. Saved to out_dir/A7-crosspatch-nie-overlay.pdf
    """
    from matplotlib.lines import Line2D

    A7_FS_TITLE  = FS_TITLE + 1
    A7_FS_LABEL  = FS_LABEL + 5
    A7_FS_TICK   = FS_TICK + 4
    A7_FS_LEGEND = FS_LEGEND + 6

    print('  [A7] Loading OLMo base (pre) from main.zip...')
    _own_zf = main_zf is None
    if _own_zf:
        main_zf = zipfile.ZipFile(MAIN_ZIP, 'r')
    pre_results = {d: _load_from_main_zip(main_zf, d, num_sample) for d in PAPER_DOMAINS}
    if _own_zf:
        main_zf.close()

    print('  [A7] Loading OLMo instruct (post, step_2000) from local...')
    post_results = {d: load_within_model_from_local(
        'OLMo-2-0425-1B-Instruct', 'allenai', 'step_2000', d, num_sample)
        for d in PAPER_DOMAINS}

    print('  [A7] Loading cross-patch directions (pre→post, post→pre)...')
    p2post_results = {d: load_cross_patch_domain('pre_to_post', d, num_sample) for d in PAPER_DOMAINS}
    p2pre_results  = {d: load_cross_patch_domain('post_to_pre', d, num_sample) for d in PAPER_DOMAINS}

    STATES_KEYS = ('bias', 'mlp', 'attn')
    STATES_LABELS_SHORT = ['States', 'MLP window', 'Attn window']
    WORDS_KEYS  = ('bias', 'pre_blank', 'blank')
    WORDS_LABELS_SHORT  = ['Effect of subject token', 'Effect of pre-target token', 'Effect of target token']

    def _nie(res, key):
        key_map = {'bias': 'bias_mean', 'mlp': 'mlp_mean', 'attn': 'attn_mean',
                   'pre_blank': 'pre_blank_mean', 'blank': 'blank_mean'}
        if res is None:
            return None
        arr = res.get(key_map[key])
        if arr is None:
            return None
        return normalized_indirect_effect(arr, res['mean_low'], res['effect_gap'])   # None if gap <= 0

    states_vals, words_vals = [], []
    for results_dict in (pre_results, post_results, p2post_results, p2pre_results):
        for d in PAPER_DOMAINS:
            res = results_dict.get(d)
            for k in STATES_KEYS:
                v = _nie(res, k)
                if v is not None:
                    states_vals.extend(v.tolist())
            for k in WORDS_KEYS:
                v = _nie(res, k)
                if v is not None:
                    words_vals.extend(v.tolist())

    if not states_vals and not words_vals:
        print('  [A7] No valid NIE data; skipping.')
        return

    def _yrange(vals):
        if not vals:
            return -0.1, 1.1
        m = (max(vals) - min(vals)) * 0.12 or 0.05
        return min(vals) - m, max(vals) + m

    states_ymin, states_ymax = _yrange(states_vals)
    words_ymin,  words_ymax  = _yrange(words_vals)

    # rows: (plot_type, within_results, within_label, cross_results, cross_label)
    ROW_DEFS = [
        ('states', post_results, 'Post within-model (solid)',
                   p2post_results, 'Pre→Post cross-patch (dashed)'),
        ('words',  post_results, 'Post within-model (solid)',
                   p2post_results, 'Pre→Post cross-patch (dashed)'),
        ('states', pre_results,  'Pre within-model (solid)',
                   p2pre_results,  'Post→Pre cross-patch (dashed)'),
        ('words',  pre_results,  'Pre within-model (solid)',
                   p2pre_results,  'Post→Pre cross-patch (dashed)'),
    ]
    letters_all = list('abcdefghijkl')

    fig, axes = plt.subplots(4, 3, figsize=(FIG_LINE_W_PER_PAN * 3, (FIG_ROW_H + 0.5) * 4))
    letter_idx = 0

    for row, (plot_type, m1, l1, m2, l2) in enumerate(ROW_DEFS):
        is_states = (plot_type == 'states')
        keys   = STATES_KEYS if is_states else WORDS_KEYS
        colors = BAR_COLORS  if is_states else WORDS_COLORS
        ymin   = states_ymin if is_states else words_ymin
        ymax   = states_ymax if is_states else words_ymax
        type_tag = 'States NIE' if is_states else 'Words NIE'

        for col, domain in enumerate(PAPER_DOMAINS):
            ax     = axes[row, col]
            letter = letters_all[letter_idx]
            letter_idx += 1
            res1 = m1.get(domain)
            res2 = m2.get(domain)

            for ki, key in enumerate(keys):
                color = colors[ki]
                nie1  = _nie(res1, key)
                nie2  = _nie(res2, key)
                if nie1 is not None:
                    ax.plot(np.arange(len(nie1)), nie1, color=color, linewidth=2.0,
                            linestyle='-', marker='o', markersize=3)
                if nie2 is not None:
                    ax.plot(np.arange(len(nie2)), nie2, color=color, linewidth=2.0,
                            linestyle='--', marker='s', markersize=3)

            ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
            ax.set_title(f'({letter}) {domain.capitalize()} — {type_tag}\n{l1} vs {l2}',
                         fontsize=A7_FS_TITLE)
            ax.set_xlabel('Layer', fontsize=A7_FS_LABEL)
            if col == 0:
                ax.set_ylabel(Y_LABEL_NIE, fontsize=A7_FS_LABEL)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(np.arange(0, 16, max(1, 16 // 8)))
            ax.tick_params(labelsize=A7_FS_TICK)
            ax.grid(alpha=0.2)

    legend_handles = []
    for ki, cond_label in enumerate(STATES_LABELS_SHORT):
        c = BAR_COLORS[ki]
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='-',
                   marker='o', markersize=4, label=f'{cond_label} — within (solid)'))
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='--',
                   marker='s', markersize=4, label=f'{cond_label} — cross (dashed)'))
    legend_handles.append(Line2D([0], [0], color='none', label=''))
    for ki, cond_label in enumerate(WORDS_LABELS_SHORT):
        c = WORDS_COLORS[ki]
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='-',
                   marker='o', markersize=4, label=f'{cond_label} — within (solid)'))
        legend_handles.append(
            Line2D([0], [0], color=c, linewidth=2, linestyle='--',
                   marker='s', markersize=4, label=f'{cond_label} — cross (dashed)'))

    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=A7_FS_LEGEND, frameon=True,
               labelspacing=1.0, handlelength=3.0, handletextpad=0.8, columnspacing=2.5)
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    _savepdf(fig, os.path.join(out_dir, 'A7-crosspatch-nie-overlay.pdf'))


# ── appendix A4: cross-patch bars (4×3) ──────────────────────────────────────

def save_appendix_A4_cross_patch_bars(all_direction_results, out_dir,
                                      out_name='A4-cross-patch-bars.pdf'):
    """
    Appendix A4: 4×3 grid for cross-patch directions.
    Row 0 Pre→Post states · Row 1 Pre→Post words ·
    Row 2 Post→Pre states · Row 3 Post→Pre words.
    Columns: gender, race, profession.
    States bars use STATES_COLORS; words bars use WORDS_COLORS.
    Single global Y-axis across all 12 panels.
    """
    pre  = all_direction_results.get('pre_to_post', {})
    post = all_direction_results.get('post_to_pre', {})

    if not pre and not post:
        print('  [A4] No cross-patch data; skipping.')
        return

    row_specs = [
        (pre,  'states', STATES_COLORS, 'Pre → Post', 'effect of states'),
        (pre,  'words',  WORDS_COLORS,  'Pre → Post', 'effect of different words'),
        (post, 'states', STATES_COLORS, 'Post → Pre', 'effect of states'),
        (post, 'words',  WORDS_COLORS,  'Post → Pre', 'effect of different words'),
    ]
    row_ylims = [
        _row_ylim(pre,  PAPER_DOMAINS, 'bias_mean', 'mlp_mean', 'attn_mean'),
        _row_ylim(pre,  PAPER_DOMAINS, 'bias_mean', 'pre_blank_mean', 'blank_mean'),
        _row_ylim(post, PAPER_DOMAINS, 'bias_mean', 'mlp_mean', 'attn_mean'),
        _row_ylim(post, PAPER_DOMAINS, 'bias_mean', 'pre_blank_mean', 'blank_mean'),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(FIG_BAR_W_PER_COL * 3, FIG_ROW_H * 4))
    _bars_grid(axes, row_specs, PAPER_DOMAINS, row_ylims)
    _states_words_legend(fig)
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    _savepdf(fig, os.path.join(out_dir, out_name))


# ── appendix A5: effect gap trajectory (3×1) ─────────────────────────────────

def save_appendix_A5_trajectory(base_stats, instruct_stats, out_dir):
    """
    Appendix A5: 3×1 effect gap trajectory for OLMo base + instruct.
    One panel per paper domain (rows); panel title = domain name only.
    OLMo base (blue solid) and instruct (orange dashed) plotted on the same x-axis.
    Y-axis shared across all 3 panels.
    """
    A5_FS_TITLE  = FS_TITLE  + 4   # 14
    A5_FS_LABEL  = FS_LABEL  + 3   # 12
    A5_FS_TICK   = FS_TICK   + 3   # 11
    A5_FS_LEGEND = FS_LEGEND + 4   # 11

    letters = ['a', 'b', 'c']

    domain_data = {}
    for domain in PAPER_DOMAINS:
        base_pts, instruct_pts = [], []
        for e in base_stats:
            s = e['domains'].get(domain)
            if s:
                base_pts.append((e['label'], s['effect_gap'], s['effect_gap'] < LOW_SIGNAL))
        for e in instruct_stats:
            s = e['domains'].get(domain)
            if s:
                instruct_pts.append((e['label'], s['effect_gap'], s['effect_gap'] < LOW_SIGNAL))
        if base_pts or instruct_pts:
            domain_data[domain] = (base_pts, instruct_pts)

    if not domain_data:
        print('  [A5] No data; skipping.')
        return

    all_gaps = [p[1] for bp, ip in domain_data.values() for p in bp + ip]
    if not all_gaps:
        return
    margin = (max(all_gaps) - min(all_gaps)) * 0.12 or 0.005
    y_min  = max(0.0, min(all_gaps) - margin)
    y_max  = max(all_gaps) + margin

    n_total = max(len(bp) + len(ip) for bp, ip in domain_data.values())
    fig_w   = max(FIG_LINE_W_PER_PAN, n_total * 1.2)
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, (FIG_ROW_H + 0.5) * 3))

    for ax, letter, domain in zip(axes, letters, PAPER_DOMAINS):
        if domain not in domain_data:
            ax.set_visible(False)
            continue
        base_pts, instruct_pts = domain_data[domain]
        n_base   = len(base_pts)
        all_pts  = base_pts + instruct_pts
        xs       = np.arange(len(all_pts))
        labels_x = [p[0] for p in all_pts]
        gaps     = np.array([p[1] for p in all_pts], dtype=float)
        low_sig  = [p[2] for p in all_pts]

        ax.plot(xs[:n_base], gaps[:n_base], 'o-',
                color=BASE_COLOR, label='Base (pre-training)',
                linewidth=2, markersize=7, zorder=3)
        if instruct_pts:
            bridge_xs = [n_base - 1] + list(xs[n_base:])
            bridge_ys = [gaps[n_base - 1]] + list(gaps[n_base:])
            ax.plot(bridge_xs, bridge_ys, 's--',
                    color=INSTRUCT_COLOR, label='Instruct fine-tuning',
                    linewidth=2, markersize=7, zorder=3)
            ax.axvline(n_base - 0.5, color='gray', linestyle='--',
                       linewidth=1.2, alpha=0.6)

        for xi, ls in enumerate(low_sig):
            if ls:
                ax.annotate('⚠', (xs[xi], gaps[xi]),
                            textcoords='offset points', xytext=(0, 6),
                            ha='center', fontsize=FS_LABEL, color=LOW_SIG_COLOR)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels_x, rotation=30, ha='right', fontsize=A5_FS_TICK)
        ax.set_xlabel('Training checkpoint', fontsize=A5_FS_LABEL)
        ax.set_ylabel('Effect gap (high − low)', fontsize=A5_FS_LABEL)
        ax.set_title(f'({letter}) {domain.capitalize()}', fontsize=A5_FS_TITLE)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='y', labelsize=A5_FS_TICK)
        ax.grid(axis='y', alpha=0.25)
        ax.axhline(0, color='black', linewidth=0.7, alpha=0.3)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=A5_FS_LEGEND, frameon=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    _savepdf(fig, os.path.join(out_dir, 'A5-trajectory.pdf'))


# ── appendix A6: NIE heatmap (layer × checkpoint) ────────────────────────────

def save_appendix_A6_heatmap(base_stats, instruct_stats, out_dir, num_sample=None):
    """
    Appendix A6: 3×3 grid of NIE heatmaps.
    Rows: states conditions (full restore, MLP-window restore, Attn-window restore).
    Columns: domains (gender, race, profession).
    X-axis: training checkpoints (base pre-training | instruct fine-tuning).
    Y-axis: layer (0–15).
    step_2000 is missing from instruct stats.json NIE arrays; loaded from local NFS.
    Colorscale shared per row (same condition, all 3 domains).
    """
    A6_FS_TITLE  = FS_TITLE  + 3   # 13
    A6_FS_LABEL  = FS_LABEL  + 4   # 13
    A6_FS_TICK   = FS_TICK   + 3   # 11
    A6_FS_CBAR   = FS_TICK   + 2   # 10

    # (display label, stats.json key, load_within_model_from_local dict key)
    # attn_score in stats = _attn.npz = restored window of Attn outputs
    # mlp_score  in stats = _mlp.npz  = restored window of MLP outputs
    conditions = [
        (STATES_LABELS[0], 'states_score', 'bias_mean'),
        (STATES_LABELS[1], 'mlp_score',    'mlp_mean'),
        (STATES_LABELS[2], 'attn_score',   'attn_mean'),
    ]

    # Load step_2000 from local NFS (full result dict with mean_low + effect_gap).
    # Other checkpoints are read from stats.json which stores raw ALP;
    # both are normalized to NIE via (alp - mean_low) / effect_gap — same as A3.
    step2000_local = {}
    for domain in PAPER_DOMAINS:
        res = load_within_model_from_local(
            'OLMo-2-0425-1B-Instruct', 'allenai', 'step_2000', domain, num_sample)
        if res is not None:
            step2000_local[domain] = res

    def _to_nie(alp_arr, mean_low, effect_gap):
        return normalized_indirect_effect(alp_arr, mean_low, effect_gap, degenerate='zero')

    def _get_from_stats(stats_list, domain, key):
        """Return [(label, nie_array)] normalizing stored raw ALP to NIE."""
        out = []
        for e in stats_list:
            d = e['domains'].get(domain, {})
            if key in d and 'mean_low' in d and 'effect_gap' in d:
                nie = _to_nie(d[key], d['mean_low'], d['effect_gap'])
                out.append((e['label'], nie))
        return out

    def _build_matrix(domain, stats_key, local_key):
        """Build (matrix, labels_x, n_base) for one domain × condition."""
        base_pts  = _get_from_stats(base_stats,    domain, stats_key)
        inst_pts  = _get_from_stats(instruct_stats, domain, stats_key)

        # Insert step_2000 in correct position (after step1400, before step2600).
        # Skip insertion if step_2000 is already present (e.g. from NPZ loading).
        inst_ordered = []
        step2000_inserted = any(l == 'step2000' for l, _ in inst_pts)
        for label, arr in inst_pts:
            inst_ordered.append((label, arr))
            if label == 'step1400' and not step2000_inserted:
                if domain in step2000_local:
                    r = step2000_local[domain]
                    inst_ordered.append(('step2000', _to_nie(r[local_key], r['mean_low'], r['effect_gap'])))
                step2000_inserted = True
        if not step2000_inserted and domain in step2000_local:
            r = step2000_local[domain]
            inst_ordered.append(('step2000', _to_nie(r[local_key], r['mean_low'], r['effect_gap'])))

        all_pts = base_pts + inst_ordered
        if not all_pts:
            return None, None, 0

        labels_x = [p[0] for p in all_pts]
        matrix   = np.stack([p[1] for p in all_pts], axis=1)  # (num_layer, n_checkpoints)
        return matrix, labels_x, len(base_pts)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    for row_i, (cond_label, stats_key, local_key) in enumerate(conditions):
        # Build matrices for all domains; collect for shared colorscale
        row_data = {}
        for domain in PAPER_DOMAINS:
            matrix, labels_x, n_base_ckpts = _build_matrix(domain, stats_key, local_key)
            if matrix is not None:
                row_data[domain] = (matrix, labels_x, n_base_ckpts)

        if not row_data:
            for ax in axes[row_i]:
                ax.set_visible(False)
            continue

        all_vals = np.concatenate([v[0].ravel() for v in row_data.values()])
        abs_max  = np.nanmax(np.abs(all_vals))
        vmin, vmax = -abs_max, abs_max

        for col_i, domain in enumerate(PAPER_DOMAINS):
            ax = axes[row_i, col_i]

            if domain not in row_data:
                ax.set_visible(False)
                continue

            matrix, labels_x, n_base_ckpts = row_data[domain]
            num_layer = matrix.shape[0]

            im = ax.imshow(
                matrix, aspect='auto', origin='lower',
                cmap='RdBu_r', vmin=vmin, vmax=vmax,
                extent=[-0.5, matrix.shape[1] - 0.5, -0.5, num_layer - 0.5])

            # Vertical separator between base pre-training and instruct fine-tuning
            ax.axvline(n_base_ckpts - 0.5, color='black', linewidth=1.5,
                       linestyle='--', alpha=0.7)

            ax.set_xticks(np.arange(len(labels_x)))
            ax.set_xticklabels(labels_x, rotation=45, ha='right', fontsize=A6_FS_TICK)
            ax.set_yticks(np.arange(0, num_layer, 2))
            ax.tick_params(axis='y', labelsize=A6_FS_TICK)

            if row_i == 0:
                ax.set_title(domain.capitalize(), fontsize=A6_FS_TITLE, fontweight='bold')

            if col_i == 0:
                ax.set_ylabel('Layer', fontsize=A6_FS_LABEL)
                ax.annotate(cond_label, xy=(-0.28, 0.5), xycoords='axes fraction',
                            rotation=90, ha='center', va='center',
                            fontsize=A6_FS_LABEL, fontweight='bold')
            else:
                ax.set_ylabel('')

            if row_i == len(conditions) - 1:
                ax.set_xlabel('Checkpoint', fontsize=A6_FS_LABEL)

            cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
            cb.ax.tick_params(labelsize=A6_FS_CBAR)
            if col_i == 2:
                cb.set_label(Y_LABEL_NIE, fontsize=A6_FS_CBAR)
            else:
                cb.set_label('')

    fig.tight_layout()
    _savepdf(fig, os.path.join(out_dir, 'A6-heatmap.pdf'))


# ── main loop ─────────────────────────────────────────────────────────────────

print(f'Opening zip: {ZIP_PATH}')
zf = zipfile.ZipFile(ZIP_PATH, 'r')
zip_names_all = zf.namelist()

all_models_stats = {}  # accumulate for cross-model plots

for model_name in (models_to_run if RUN_BARS or RUN_DELTA or RUN_COMPARE else []):
    cfg = MODEL_CONFIGS[model_name]
    org = cfg['org']

    model_out_dir  = os.path.join(PLOTS_BASE, model_name)
    all_ckpt_stats = []

    # cross-checkpoint accumulator: domain → list of (label, r1, r2, r3, nl)
    cross_states = {d: [] for d in BIAS_TYPES}
    cross_words  = {d: [] for d in BIAS_TYPES}

    for checkpoint, ckpt_label in cfg['checkpoints']:
        print(f'\n=== {model_name}  [{ckpt_label}] ===')
        out_ckpt_dir = os.path.join(PLOTS_BASE, model_name, ckpt_label)

        states_data = {}
        words_data  = {}
        ckpt_stats  = {'checkpoint': checkpoint, 'label': ckpt_label, 'domains': {}}

        for domain in domains_to_run:
            print(f'  {domain}')

            local_dir  = local_cases_dir(model_name, org, checkpoint, domain)
            zip_prefix = zip_cases_prefix(org, model_name, checkpoint, domain)
            zip_has_data = any(n.startswith(zip_prefix) and n.endswith('.npz')
                               for n in zip_names_all)
            use_local  = (args.source == 'local') or \
                         (args.source == 'auto' and os.path.isdir(local_dir)
                          and not zip_has_data)

            if use_local:
                all_local    = sorted(os.listdir(local_dir))
                single_b, attn_b, mlp_b = partition_names(all_local)
                single_b, attn_b, mlp_b = subsample_aligned(single_b, attn_b, mlp_b, args.num_sample)
                single_items = [os.path.join(local_dir, b) for b in single_b]
                attn_items   = [os.path.join(local_dir, b) for b in attn_b]
                mlp_items    = [os.path.join(local_dir, b) for b in mlp_b]
                loader       = load_npz_local
                print(f'    [local NFS]')
            else:
                prefix    = zip_cases_prefix(org, model_name, checkpoint, domain)
                all_names = [n for n in zip_names_all
                             if n.startswith(prefix) and n.endswith('.npz')]
                if not all_names:
                    print(f'    No data in zip for {prefix}; skipping.')
                    continue
                basenames    = [os.path.basename(n) for n in all_names]
                single_b, attn_b, mlp_b = partition_names(basenames)
                single_b, attn_b, mlp_b = subsample_aligned(single_b, attn_b, mlp_b, args.num_sample)
                name_map     = {os.path.basename(n): n for n in all_names}
                single_items = [name_map[b] for b in single_b if b in name_map]
                attn_items   = [name_map[b] for b in attn_b   if b in name_map]
                mlp_items    = [name_map[b] for b in mlp_b    if b in name_map]
                loader       = lambda p: load_npz_zip(zf, p)
                print(f'    [zip]')

            print(f'    single={len(single_items)}, attn={len(attn_items)}, mlp={len(mlp_items)}')
            if not single_items:
                print('    No single-state files; skipping.')
                continue

            expected_model = f'{org}/{model_name}'
            expected = {
                'score_metric': SCORE_METRIC,
                'source_model': expected_model,
                'target_model': expected_model,
                'direction': f'{expected_model} -> {expected_model}',
            }
            if not _validate_result_group(
                    (single_items, attn_items, mlp_items), loader,
                    expected, 'within_model'):
                continue

            try:
                num_layer = loader(single_items[0])['scores'].shape[-1]
            except Exception as ex:
                print(f'    Cannot read sample: {ex}; skipping.')
                continue

            bias_mean, pre_blank_mean, blank_mean, n_cases, mean_high, mean_low = \
                collect_scores(single_items, loader)
            attn_mean, _, _, _, _, _ = collect_scores(attn_items, loader)
            mlp_mean,  _, _, _, _, _ = collect_scores(mlp_items,  loader)

            if bias_mean is None:
                print('    No valid scores; skipping.')
                continue

            zero           = np.zeros(num_layer)
            attn_mean      = attn_mean      if attn_mean      is not None else zero
            mlp_mean       = mlp_mean       if mlp_mean       is not None else zero
            pre_blank_mean = pre_blank_mean if pre_blank_mean is not None else zero
            blank_mean     = blank_mean     if blank_mean     is not None else zero

            # ── per-checkpoint individual PDFs ─────────────────────────────
            if RUN_BARS:
                save_individual(
                    bias_mean, mlp_mean, attn_mean,
                    STATES_LABELS, BAR_COLORS, num_layer,
                    f'{domain.title()} bias effect of states ({model_name}  [{ckpt_label}])',
                    os.path.join(out_ckpt_dir, f'{domain}-states.pdf'),
                )
                save_individual(
                    bias_mean, pre_blank_mean, blank_mean,
                    WORDS_LABELS, BAR_COLORS, num_layer,
                    f'{domain.title()} bias effect of different words ({model_name}  [{ckpt_label}])',
                    os.path.join(out_ckpt_dir, f'{domain}-words.pdf'),
                )

            states_data[domain] = (bias_mean, mlp_mean,       attn_mean,  num_layer)
            words_data[domain]  = (bias_mean, pre_blank_mean, blank_mean, num_layer)

            # accumulate for cross-checkpoint figures
            cross_states[domain].append((ckpt_label, bias_mean, mlp_mean, attn_mean, num_layer))
            cross_words[domain].append( (ckpt_label, bias_mean, pre_blank_mean, blank_mean, num_layer))

            # accumulate for stats/report
            ckpt_stats['domains'][domain] = {
                'n_cases':           n_cases,
                'num_layers':        int(num_layer),
                'mean_high':         round(mean_high, 6),
                'mean_low':          round(mean_low,  6),
                'effect_gap':        round(mean_high - mean_low, 6),
                'states_score':      bias_mean.tolist(),
                'attn_score':        attn_mean.tolist(),
                'mlp_score':         mlp_mean.tolist(),
                'pre_blank_score':   pre_blank_mean.tolist(),
                'blank_score':       blank_mean.tolist(),
                'peak_layer_states': int(np.argmax(bias_mean)),
                'peak_layer_mlp':    int(np.argmax(mlp_mean)),
                'peak_layer_attn':   int(np.argmax(attn_mean)),
                'top3_states':       _top_layers(bias_mean),
                'top3_mlp':          _top_layers(mlp_mean),
                'top3_attn':         _top_layers(attn_mean),
            }

        # ── composite per checkpoint ────────────────────────────────────────
        if RUN_BARS:
            if states_data:
                save_composite(states_data, 'states', model_name, ckpt_label, out_ckpt_dir)
            if words_data:
                save_composite(words_data,  'words',  model_name, ckpt_label, out_ckpt_dir)
            if states_data and words_data:
                save_composite_all(states_data, words_data, model_name, ckpt_label, out_ckpt_dir)

        all_ckpt_stats.append(ckpt_stats)

    # ── cross-checkpoint figures (one per domain, at model level) ───────────
    if RUN_BARS:
        print(f'\n  Generating cross-checkpoint figures for {model_name}...')
        for domain in domains_to_run:
            if cross_states[domain]:
                save_cross_checkpoint(cross_states[domain], 'states', model_name, domain, model_out_dir)
            if cross_words[domain]:
                save_cross_checkpoint(cross_words[domain],  'words',  model_name, domain, model_out_dir)

    # ── stats.json + report.md ──────────────────────────────────────────────
    if all_ckpt_stats:
        save_stats_and_report(model_name, all_ckpt_stats, model_out_dir)
        all_models_stats[model_name] = all_ckpt_stats

    # ── bias delta (per-model) ───────────────────────────────────────────────
    if RUN_DELTA and all_ckpt_stats:
        save_bias_delta(all_ckpt_stats, model_name, model_out_dir)

# ── base vs instruct (cross-model) ────────────────────────────────────────────
BASE     = 'OLMo-2-0425-1B'
INSTRUCT = 'OLMo-2-0425-1B-Instruct'
if RUN_COMPARE and BASE in all_models_stats and INSTRUCT in all_models_stats:
    compare_dir = os.path.join(PLOTS_BASE, 'compare')
    os.makedirs(compare_dir, exist_ok=True)
    print(f'\n  Generating base vs instruct comparison → plots/compare/')
    save_bias_trajectory(
        all_models_stats[BASE],
        all_models_stats[INSTRUCT],
        compare_dir,
    )
    save_base_vs_instruct(
        all_models_stats[BASE],
        all_models_stats[INSTRUCT],
        compare_dir,
    )

# ── cross-patch execution block ───────────────────────────────────────────────
if RUN_CROSS_PATCH:
    print('\n=== Cross-patch plots ===')
    cross_patch_out = os.path.join(PLOTS_BASE, 'cross_patch')
    os.makedirs(cross_patch_out, exist_ok=True)

    all_direction_results = {}  # olmo_1b results — {direction_key: {domain: result_dict}}

    for cp_family in cp_families_to_run:
        fam_label = CROSS_PATCH_FAMILIES[cp_family]['label']
        # olmo_1b keeps its original output paths and titles (no family suffix);
        # newer families get a plots/cross_patch/{family}/ subtree
        is_olmo = cp_family == 'olmo_1b'
        fam_out = cross_patch_out if is_olmo else os.path.join(cross_patch_out, cp_family)
        title_label = '' if is_olmo else fam_label
        print(f'\n  Family: {fam_label}')

        fam_direction_results = {}  # {direction_key: {domain: result_dict}}

        for direction_key in directions_to_run:
            cfg_cp = CROSS_PATCH_CONFIGS[direction_key]
            print(f'\n  Direction: {cfg_cp["label"]}  ({cfg_cp["desc"]})')

            dir_out = os.path.join(fam_out, direction_key)
            os.makedirs(dir_out, exist_ok=True)

            domain_results = {}
            for domain in domains_to_run:
                print(f'  {domain}')
                res = load_cross_patch_domain(direction_key, domain, args.num_sample,
                                              family=cp_family)
                if res is not None:
                    domain_results[domain] = res

            if domain_results:
                save_cross_patch_direction(direction_key, domain_results, dir_out,
                                           family_label=title_label)
                fam_direction_results[direction_key] = domain_results

        # comparison plot — requires both directions to have results
        if len(fam_direction_results) >= 2:
            print('\n  Generating direction-comparison plots...')
            save_cross_patch_comparison(fam_direction_results, domains_to_run, fam_out,
                                        family_label=title_label)
        elif len(fam_direction_results) == 1:
            print('\n  Only one direction has data — skipping comparison plots.')

        if is_olmo:
            all_direction_results = fam_direction_results

    # 4-panel comparison: within-model last checkpoints + both cross-patch directions
    # Base: OLMo-2-0425-1B main checkpoint (from main.zip)
    # Instruct: OLMo-2-0425-1B-Instruct step_2000 main checkpoint (from local NFS)
    if all_direction_results:
        print('\n  Loading within-model main checkpoints for 4-panel plots...')
        within_model_panels = {}
        _main_zf4 = zipfile.ZipFile(MAIN_ZIP, 'r')
        for domain in domains_to_run:
            print(f'  {domain}')
            base_res = _load_from_main_zip(_main_zf4, domain, args.num_sample)
            inst_res = load_within_model_from_local(
                INSTRUCT, 'allenai', 'step_2000', domain, args.num_sample)
            within_model_panels[domain] = {'base_last': base_res, 'inst_last': inst_res}
        _main_zf4.close()
        print('\n  Generating 4-panel plots (main / step2000)...')
        _main_body_out = os.path.join(PLOTS_BASE, 'main_body')
        save_cross_patch_4panel(within_model_panels, all_direction_results,
                                domains_to_run, cross_patch_out,
                                inst_label='OLMo-2-0425-1B-Instruct\n(post)',
                                file_prefix='4panel-step2000',
                                extra_out_dir=_main_body_out)

# ── NPZ-based checkpoint loader (bypasses stats.json) ────────────────────────

def _load_checkpoints_from_npz(model_name, num_sample=None):
    """
    Build checkpoint stats list from NPZ files, bypassing stats.json entirely.
    Returns a list in the same format as stats.json['checkpoints'].
    Source priority: local NFS > main.zip (OLMo-2-0425-1B 'main') > results.zip.
    """
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        print(f'  [NPZ] Unknown model: {model_name}')
        return None
    org     = cfg['org']
    entries = []
    for ck_dir, label in cfg.get('checkpoints', []):
        entry = {'label': label, 'checkpoint': ck_dir, 'domains': {}}
        for domain in BIAS_TYPES:
            res = load_within_model_from_local(model_name, org, ck_dir, domain, num_sample)
            if res is None and model_name == 'OLMo-2-0425-1B' and ck_dir == 'main':
                _mzf = zipfile.ZipFile(MAIN_ZIP, 'r')
                res  = _load_from_main_zip(_mzf, domain, num_sample)
                _mzf.close()
            if res is None:
                res = load_within_model_from_zip(
                    zf, zip_names_all, model_name, org, ck_dir, domain, num_sample)
            if res is None:
                continue
            entry['domains'][domain] = {
                'states_score':      res['bias_mean'].tolist(),
                'pre_blank_score':   res['pre_blank_mean'].tolist(),
                'blank_score':       res['blank_mean'].tolist(),
                'mlp_score':         res['mlp_mean'].tolist(),
                'attn_score':        res['attn_mean'].tolist(),
                'mean_high':         float(res['mean_high']),
                'mean_low':          float(res['mean_low']),
                'effect_gap':        float(res['effect_gap']),
                'n_cases':           res['n_cases'],
                'num_layers':        res['num_layer'],
                'peak_layer_states': int(np.argmax(res['bias_mean'])),
                'peak_layer_mlp':    int(np.argmax(res['mlp_mean'])),
                'peak_layer_attn':   int(np.argmax(res['attn_mean'])),
            }
        entries.append(entry)
    n_loaded = sum(len(e['domains']) for e in entries)
    print(f'  [NPZ] {model_name}: {n_loaded} domain entries across {len(entries)} checkpoints')
    return entries


# ── appendix figures ──────────────────────────────────────────────────────────
if RUN_APPENDIX:
    print('\n=== Appendix figures ===')
    appendix_out = os.path.join(PLOTS_BASE, 'appendix')
    os.makedirs(appendix_out, exist_ok=True)

    _app_main_zf = zipfile.ZipFile(MAIN_ZIP, 'r')

    print('\n  A1: OLMo base + instruct bar charts (4×3)...')
    save_appendix_A1_olmo_bars(appendix_out, args.num_sample, main_zf=_app_main_zf)

    print('\n  A2: Pythia bar charts (2×3)...')
    save_appendix_A2_pythia_bars(appendix_out, args.num_sample)

    print('\n  A3: NIE line plots (2×3)...')
    save_appendix_A3_nie_lines(appendix_out, args.num_sample, main_zf=_app_main_zf)

    print('\n  NIE overlay (main body, gender only)...')
    main_body_out = os.path.join(PLOTS_BASE, 'main_body')
    os.makedirs(main_body_out, exist_ok=True)
    save_main_body_nie_overlay(main_body_out, args.num_sample, main_zf=_app_main_zf)

    _app_main_zf.close()

    print('\n  A4: Cross-patch bar charts (4×3)...')
    for _fam in cp_families_to_run:
        # olmo_1b keeps the original A4-cross-patch-bars.pdf name; newer
        # families get a -{family} suffix
        _fname = ('A4-cross-patch-bars.pdf' if _fam == 'olmo_1b'
                  else f'A4-cross-patch-bars-{_fam}.pdf')
        print(f'    family: {_fam}')
        _app_cp = {}
        for _dk in ('pre_to_post', 'post_to_pre'):
            _dr = {}
            for _dom in PAPER_DOMAINS:
                print(f'    {_dk} / {_dom}')
                _res = load_cross_patch_domain(_dk, _dom, args.num_sample, family=_fam)
                if _res is not None:
                    _dr[_dom] = _res
            if _dr:
                _app_cp[_dk] = _dr
        save_appendix_A4_cross_patch_bars(_app_cp, appendix_out, out_name=_fname)

    print('\n  A5: Trajectory plots (1×3)...')
    _base_traj     = _load_checkpoints_from_npz('OLMo-2-0425-1B',          args.num_sample)
    _instruct_traj = _load_checkpoints_from_npz('OLMo-2-0425-1B-Instruct', args.num_sample)
    if _base_traj and _instruct_traj:
        save_appendix_A5_trajectory(_base_traj, _instruct_traj, appendix_out)
    else:
        print('  [A5] Skipping.')

    print('\n  A6: NIE heatmap — layer × checkpoint (3 conditions × 3 domains)...')
    if _base_traj and _instruct_traj:
        save_appendix_A6_heatmap(_base_traj, _instruct_traj, appendix_out, args.num_sample)
    else:
        print('  [A6] Skipping.')

# ── cross-patch NIE overlay (within-model vs cross-patch, 4×3) ────────────────
if RUN_CP_NIE:
    print('\n=== Cross-patch NIE overlay (within-model vs cross-patch) ===')
    cp_nie_out = os.path.join(PLOTS_BASE, 'appendix')
    os.makedirs(cp_nie_out, exist_ok=True)
    save_crosspatch_nie_overlay(cp_nie_out, args.num_sample)

    cp_mb_out = os.path.join(PLOTS_BASE, 'main_body')
    os.makedirs(cp_mb_out, exist_ok=True)
    save_main_body_pre_post_crosspatch(cp_mb_out, args.num_sample)

zf.close()
print('\nDone.')
