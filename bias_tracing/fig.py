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
import functools
import math
import numpy as np
from tqdm import tqdm
import matplotlib
from matplotlib.patches import Patch
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_utils import (
    ZIP_PATH, MAIN_ZIP, PLOTS_BASE, USE_PKG_RESULTS, BASE_MODEL, plot_dir, paper_dir, disk_cache,
    MODEL_CONFIGS, BIAS_TYPES, PAPER_DOMAINS,
    CROSS_PATCH_BASE, CROSS_PATCH_CONFIGS, CROSS_PATCH_FAMILIES,
    cross_patch_cases_dir,
    STATES_LABELS, WORDS_LABELS, BAR_COLORS, STATES_COLORS, WORDS_COLORS,
    LOW_SIGNAL, Y_LABEL_BARS, Y_LABEL_NIE, FAMILY_PLOT_DIRS,
    FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    PAPER_WIDTH, PAPER_COL_WIDTH, PAPER_PANEL_ASPECT, PAPER_COL_ASPECT, PAPER_LEGEND_ROW_H,
    PAPER_FS_TITLE, PAPER_FS_LABEL, PAPER_FS_TICK, PAPER_FS_LEGEND,
    LABEL_PRE, LABEL_POST, LABEL_WITHIN, LABEL_CROSS, STYLE_FIRST, STYLE_SECOND, LABEL_P2POST, LABEL_POST2PRE,
    STYLE_PRE, STYLE_POST, STYLE_P2POST, STYLE_POST2PRE, CKPT_TICKS,
    PAPER_ROW_H, PAPER_TRAJ_ROW_H, DOMAIN_COLORS, REF_CLEAN, REF_CORRUPT, TABLES_DIR,
    FIG_BAR_W_SINGLE, FIG_BAR_H_SINGLE, FIG_BAR_W_PER_COL,
    FIG_GRID_W_PER_COL, FIG_ROW_H, FIG_LINE_W_PER_PAN, FIG_TRAJ_H,
    BASE_COLOR, INSTRUCT_COLOR, LOW_SIG_COLOR, LOW_SIG_BG,
    local_cases_dir, zip_cases_prefix, partition_names, subsample_aligned,
    load_npz_local, load_npz_zip,
    SCORE_METRIC, validate_score_files, collect_scores,
    _draw_bars, _savepdf, share_ylim, normalized_indirect_effect, signed_gap_reliable,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

PLOT_CHOICES = ['bars', 'delta', 'compare', 'cross_patch', 'appendix', 'cp_nie', 'main']

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
RUN_MAIN        = 'main'        in _plots

models_to_run      = [args.model]     if args.model     else list(MODEL_CONFIGS.keys())
domains_to_run     = [args.bias]      if args.bias       else BIAS_TYPES
directions_to_run  = [args.direction] if args.direction  else list(CROSS_PATCH_CONFIGS.keys())
cp_families_to_run = [args.cp_family] if args.cp_family  else list(CROSS_PATCH_FAMILIES.keys())

def _paper(legacy_dir, family=None):
    """Paper figure folder: new_plots/main/ (family=None) or new_plots/appendix/<family>/."""
    if USE_PKG_RESULTS:
        return paper_dir(family)
    os.makedirs(legacy_dir, exist_ok=True)
    return legacy_dir


def _out(legacy_dir, family, score, category):
    """Output folder: new_plots/<family>/<score>/<category>/ in package mode, else the old one."""
    if USE_PKG_RESULTS:
        return plot_dir(family, score, category)
    os.makedirs(legacy_dir, exist_ok=True)
    return legacy_dir


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
    share_ylim(axes)
    plt.tight_layout()
    _savepdf(fig, os.path.join(out_dir, f'composite-{plot_type}.pdf'))

def save_composite_all(states_data, words_data, model_name, ckpt_label, out_dir):
    """2×N grid — top row=states, bottom row=words, columns=domains."""
    domains = [d for d in BIAS_TYPES if d in states_data and d in words_data]
    if not domains:
        return
    fig, axes = plt.subplots(2, len(domains),
                             figsize=(FIG_BAR_W_PER_COL * len(domains), FIG_ROW_H * 2))
    if len(domains) == 1:
        axes = axes[:, np.newaxis]  # matplotlib squeezes a size-1 axis away by default;
        # same fix save_cross_patch_direction's identical 2xN grid already uses below.
    fig.suptitle(f'All bias domains — {model_name}  [{ckpt_label}]',
                 fontsize=FS_SUPTITLE, fontweight='bold')
    for col, domain in enumerate(domains):
        r1, r2, r3, nl = states_data[domain]
        _draw_bars(axes[0, col], r1, r2, r3, STATES_LABELS, BAR_COLORS, nl,
                   'Layer', Y_LABEL_BARS, f'{domain.title()} — states')
        r1, r2, r3, nl = words_data[domain]
        _draw_bars(axes[1, col], r1, r2, r3, WORDS_LABELS, BAR_COLORS, nl,
                   'Layer', Y_LABEL_BARS, f'{domain.title()} — words')
    share_ylim(axes)
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
    # plt.subplots(nrows, ncols) with ncols hardcoded to 2 never returns a bare Axes,
    # even when n == 1 -- it's always at least a (2,) array (one real subplot, one
    # empty one hidden below). `if n > 1 else [axes]` assumed the bare-Axes case could
    # happen here; it can't, so that branch wrapped a (2,)-array in another list and
    # crashed the first `.bar()` call downstream on whichever (model, domain) happened
    # to have exactly one item to plot.
    axes_flat = np.atleast_1d(axes).flatten()
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
        f'only that one state is restored. Reported here as NIE (normalized by the clean–corrupted gap),',
        f'computed on the **signed** stereo-minus-anti score, not its absolute value -- so a layer that',
        f'pushes the model toward the anti-stereotype is distinguishable from one that restores the',
        f'stereotype, rather than both scoring as equally "recovered". "n/a" means this result predates',
        f'the signed score and only the absolute-value chain is available.',
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
        f'| **Effect gap** | High − Low, absolute chain — reduction in separation after corruption; < 0.03 = low-signal |',
        f'| **Peak All/MLP/Attn** | Layer with highest raw patched score under each restore condition |',
        f'| **NIE L0/L-mid/L-last** | Normalized indirect effect (signed chain) at layer 0 / middle / final layer |',
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
            gap  = s['effect_gap']          # abs chain, for this report only; the paper flags on the signed gap
            flag = ' ⚠' if gap < LOW_SIGNAL else ''
            nl   = s['num_layers']
            mid  = nl // 2
            gap_s, low_s = s.get('effect_gap_signed'), s.get('mean_low_signed')
            # signed_gap_reliable, not just "is not None": a pooled signed gap can be
            # nonzero but still small enough to amplify per-layer noise into large,
            # spurious-looking NIE values (see its docstring) -- reject those too, not
            # just the literal-zero denominator normalized_indirect_effect itself guards.
            if low_s is not None and signed_gap_reliable(gap_s):
                nie_l0   = normalized_indirect_effect(s['states_score_signed'][0],   low_s, gap_s, degenerate='zero', signed=True)
                nie_lmid = normalized_indirect_effect(s['states_score_signed'][mid], low_s, gap_s, degenerate='zero', signed=True)
                nie_last = normalized_indirect_effect(s['states_score_signed'][-1],  low_s, gap_s, degenerate='zero', signed=True)
                nie_cols = f"{nie_l0:+.2f} | {nie_lmid:+.2f} | {nie_last:+.2f}"
            else:
                nie_cols = "n/a | n/a | n/a"
            lines.append(
                f"| {e['label']} | {domain}{flag} | {s['n_cases']} | {gap:.4f} "
                f"| {s['peak_layer_states']} | {s['peak_layer_mlp']} | {s['peak_layer_attn']} "
                f"| {nie_cols} |"
            )

    lines += [
        f'',
        f'---',
        f'',
        f'## Normalized Indirect Effect (NIE) by layer — States (full restore)',
        f'',
        f'NIE = (restoration_score - low_score) / (high_score - low_score), all three on the',
        f'**signed** chain (stereo-minus-anti, not its absolute value).',
        f'',
        f'- **NIE > 0**: restoring this (subject token, layer) recovers some of the clean model\'s own',
        f'  preference (whichever direction that preference happened to be).',
        f'- **NIE = 1**: full recovery of that preference.',
        f'- **NIE < 0**: restoring this position pushes the prediction *away* from the clean model\'s',
        f'  own preference — either the position pulls toward the opposite (stereo vs. anti) direction,',
        f'  or the model\'s internal state has become inconsistent from partially restoring only one',
        f'  position. The two aren\'t distinguished by this number alone.',
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
                flag = '  ⚠ low-signal' if s['effect_gap'] < LOW_SIGNAL else ''
                gap_s, low_s = s.get('effect_gap_signed'), s.get('mean_low_signed')
                if low_s is not None and signed_gap_reliable(gap_s):
                    nie_vals = normalized_indirect_effect(s['states_score_signed'], low_s, gap_s, degenerate='zero', signed=True)
                    vals = ' | '.join(f'{v:+.2f}' for v in nie_vals)
                else:
                    vals = ' | '.join(['n/a'] * nl)
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
        # plt.subplots(nrows, ncols) with ncols hardcoded to 2 never returns a bare Axes,
        # even when n == 1 -- it's always at least a (2,) array (one real subplot, one
        # empty one hidden below). `if n > 1 else [axes]` assumed the bare-Axes case could
        # happen here; it can't, so that branch wrapped a (2,)-array in another list and
        # crashed the first `.bar()` call downstream on whichever (model, domain) happened
        # to have exactly one item to plot.
        axes_flat = np.atleast_1d(axes).flatten()

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
      1. Effect gap (high − low) on the abs chain (supporting plot; the paper's A5 uses the signed gap).
      2. Embedding layer contribution — fraction of effect gap recovered at L0
         = (states_score_signed[0] − mean_low_signed) / effect_gap_signed. Signed chain, so a
         checkpoint whose L0 restoration pushes toward the anti-stereotype shows as negative
         rather than being folded in as "recovered". Scale-invariant across checkpoints.
         NaN when the result predates scores_signed.
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
            gap_s, low_s = s.get('effect_gap_signed'), s.get('mean_low_signed')
            frac_l0 = (normalized_indirect_effect(s['states_score_signed'][0], low_s, gap_s, degenerate='nan', signed=True)
                       if low_s is not None and signed_gap_reliable(gap_s) else float('nan'))
            base_pts.append((e['label'], gap, frac_l0, raw_l0, gap < LOW_SIGNAL))

        for e in instruct_stats:
            s = e['domains'].get(domain)
            if not s:
                continue
            gap    = s['effect_gap']
            raw_l0 = s['states_score'][0]
            gap_s, low_s = s.get('effect_gap_signed'), s.get('mean_low_signed')
            frac_l0 = (normalized_indirect_effect(s['states_score_signed'][0], low_s, gap_s, degenerate='nan', signed=True)
                       if low_s is not None and signed_gap_reliable(gap_s) else float('nan'))
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
             'NIE at L0 (signed)',
             'NIE at L0 — normalized indirect effect at the first transformer layer, signed chain\n'
             '= (Patched − Corrupted) / (Clean − Corrupted), stereo-minus-anti (not abs)'),
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
        # plt.subplots(nrows, ncols) with ncols hardcoded to 2 never returns a bare Axes,
        # even when n == 1 -- it's always at least a (2,) array (one real subplot, one
        # empty one hidden below). `if n > 1 else [axes]` assumed the bare-Axes case could
        # happen here; it can't, so that branch wrapped a (2,)-array in another list and
        # crashed the first `.bar()` call downstream on whichever (model, domain) happened
        # to have exactly one item to plot.
        axes_flat = np.atleast_1d(axes).flatten()

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


def _finalize_result_dict(single_items, attn_items, mlp_items, loader, num_layer):
    """Shared tail for every within-model/cross-patch loader below: collect both the
    abs chain (ALP, always) and the signed chain (NIE, when the result files carry
    scores_signed), zero-fill missing attn/mlp/word arrays, and assemble the one result
    dict shape every figure function reads. Was duplicated near-verbatim across four
    loaders before this; pulling it out means every figure gets the signed chain the
    same way, from one place, rather than four copies that could drift apart.

    Returns None if the single-state files have no valid (abs) scores at all.
    """
    bias_mean, pre_blank_mean, blank_mean, n_cases, mean_high, mean_low = \
        collect_scores(single_items, loader)
    attn_mean, _, _, _, _, _ = collect_scores(attn_items, loader)
    mlp_mean,  _, _, _, _, _ = collect_scores(mlp_items,  loader)
    if bias_mean is None:
        return None

    bias_mean_s, pre_blank_mean_s, blank_mean_s, _, mean_high_s, mean_low_s = \
        collect_scores(single_items, loader, signed=True)
    attn_mean_s, attn_pre_s, _, _, _, _ = collect_scores(attn_items, loader, signed=True)
    mlp_mean_s,  mlp_pre_s,  _, _, _, _ = collect_scores(mlp_items,  loader, signed=True)
    have_signed = bias_mean_s is not None

    zero           = np.zeros(num_layer)
    attn_mean      = attn_mean      if attn_mean      is not None else zero
    mlp_mean       = mlp_mean       if mlp_mean       is not None else zero
    pre_blank_mean = pre_blank_mean if pre_blank_mean is not None else zero
    blank_mean     = blank_mean     if blank_mean     is not None else zero
    effect_gap     = mean_high - mean_low

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
        # signed chain -- NIE only. None (not zero-filled) when scores_signed isn't in
        # the result files, so callers can tell "no signal" from "no data" apart -- the
        # same distinction _finalize_result_dict's caller made for the abs chain via
        # bias_mean is None above.
        'bias_mean_signed':      bias_mean_s      if have_signed else None,
        'pre_blank_mean_signed': pre_blank_mean_s if have_signed else None,
        'blank_mean_signed':     blank_mean_s     if have_signed else None,
        'attn_mean_signed':      attn_mean_s      if have_signed else None,
        'attn_pre_signed':       attn_pre_s       if have_signed else None,
        'mlp_pre_signed':        mlp_pre_s        if have_signed else None,
        'mlp_mean_signed':       mlp_mean_s       if have_signed else None,
        'mean_high_signed':      mean_high_s      if have_signed else None,
        'mean_low_signed':       mean_low_s       if have_signed else None,
        'effect_gap_signed':     (mean_high_s - mean_low_s) if have_signed else None,
    }


@functools.lru_cache(maxsize=None)   # each figure reloads the same runs; read them once
@disk_cache(lambda direction_key, domain, num_sample=None, family='olmo_1b':
            cross_patch_cases_dir(family, direction_key, domain))
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

    result = _finalize_result_dict(single_items, attn_items, mlp_items, load_npz_local, num_layer)
    if result is None:
        print('    No valid scores; skipping.')
    return result


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
        if _low_sig(res):
            ax.set_facecolor(LOW_SIG_BG)
            ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                    fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

    # ── individual PDFs per domain ────────────────────────────────────────────
    for domain, res in domain_results.items():
        nl       = res['num_layer']
        sig_flag = ' [low-signal]' if _low_sig(res) else ''

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
        share_ylim(axes)
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
    share_ylim(axes)
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
                                 res['num_layer'], _low_sig(res)))

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
    if USE_PKG_RESULTS:
        return load_within_model_from_local(model_name, org, checkpoint, domain, num_sample)
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

    return _finalize_result_dict(single_items, attn_items, mlp_items, loader, num_layer)


@functools.lru_cache(maxsize=None)
@disk_cache(lambda model_name, org, checkpoint, domain, num_sample=None:
            local_cases_dir(model_name, org, checkpoint, domain))
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

    return _finalize_result_dict(single_items, attn_items, mlp_items, loader, num_layer)


def save_cross_patch_4panel(within_model_panels, all_direction_results, domains, out_dir,
                            inst_label='OLMo Instruct\n(step2600)', file_prefix='4panel',
                            base_label='OLMo-2-0425-1B\n(pre)',
                            extra_out_dir=None, within_out_dir=None):
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
        ('base_last',    base_label,               'within'),
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
        if _low_sig(res):
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
                os.path.join(within_out_dir or out_dir, f'{file_prefix}-{domain}-{plot_type}-within.pdf'))

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
    if USE_PKG_RESULTS:
        return load_within_model_from_local(BASE_MODEL, 'allenai', 'main', domain, num_sample)
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
    return _finalize_result_dict(single_items, attn_items, mlp_items, loader, num_layer)


def _load_family_pair(family, domains, num_sample=None, main_zf=None):
    """Within-model base and instruct results for one CROSS_PATCH_FAMILIES entry (or 'pythia')."""
    if family == 'pythia':   # no instruct model
        base = {d: load_within_model_from_local('pythia-1b', 'EleutherAI', 'step143000', d, num_sample)
                for d in domains}
        return base, {}, 'Pythia-1b (step143000)', None
    fam = CROSS_PATCH_FAMILIES[family]
    (b_org, b_name), (i_org, i_name) = (fam[k].split('/') for k in ('base_model', 'instruct_model'))
    if family == 'olmo_1b':
        own = main_zf is None and not USE_PKG_RESULTS
        if own:
            main_zf = zipfile.ZipFile(MAIN_ZIP, 'r')
        base = {d: _load_from_main_zip(main_zf, d, num_sample) for d in domains}
        if own:
            main_zf.close()
        inst_ckpt = 'step_2000'
    else:
        base = {d: load_within_model_from_local(b_name, b_org, 'main', d, num_sample) for d in domains}
        inst_ckpt = 'main'
    inst = {d: load_within_model_from_local(i_name, i_org, inst_ckpt, d, num_sample) for d in domains}
    return base, inst, b_name, i_name


def _n_layers(*results):
    return max((r['num_layer'] for r in results if r), default=16)


def _layer_ticks(n, n_ticks=8):
    return np.arange(0, n, max(1, n // n_ticks))


# ── paper figures: shared layout (all sizes and fonts come from plot_utils) ───

def _paper_fig(n_rows, n_cols, legend_rows=1, width=PAPER_WIDTH, row_h=None):
    """Paper figure of n_rows x n_cols panels at the common print scale, with room for a legend below.
    row_h fixes the panel height (one-row main-body figures); otherwise it follows the panel aspect."""
    aspect = PAPER_COL_ASPECT if width == PAPER_COL_WIDTH else PAPER_PANEL_ASPECT
    height = n_rows * (row_h or (width / n_cols) * aspect) + legend_rows * PAPER_LEGEND_ROW_H + 0.3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)
    return fig, axes


def _paper_legend(fig, handles, ncol):
    """One legend below all panels; then fit the panels above it."""
    leg = fig.legend(handles=handles, loc='lower center', ncol=ncol, fontsize=PAPER_FS_LEGEND, frameon=True,
                     handlelength=2.2, columnspacing=1.4)
    fig.canvas.draw()   # reserve exactly the legend's height, so no blank band is left above it
    leg_h = leg.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    fig.tight_layout(rect=[0, (leg_h + 0.08) / fig.get_figheight(), 1, 1])


def _panel(ax, title, n_lay=None, ylabel='', n_cols=3):
    """Panel label, axis labels and ticks in the paper style."""
    ax.set_title(title, fontsize=PAPER_FS_TITLE)
    ax.set_xlabel('Layer', fontsize=PAPER_FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=PAPER_FS_LABEL)
    if n_lay:
        ax.set_xticks(_layer_ticks(n_lay, 4 if n_cols >= 4 else 8))
    ax.tick_params(labelsize=PAPER_FS_TICK)


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


def _bars_grid(axes, row_specs, paper_domains, signed=False):
    """Bar charts on a rows x domains grid. row_specs: (results_by_domain, 'states' | 'words') per row.
    signed=False: ALP; signed=True: LPD, with the clean and corrupted levels as reference lines.
    Panels are labelled "(a) Gender"; what each row shows goes in the caption."""
    letter = 0
    for row, (results_dict, plot_type) in enumerate(row_specs):
        colors, labels = (STATES_COLORS, STATES_LABELS) if plot_type == 'states' else (WORDS_COLORS, WORDS_LABELS)
        for col, domain in enumerate(paper_domains):
            ax, res = axes[row, col], results_dict.get(domain)
            title = f'({"abcdefghijklmnopqrstuvwxyz"[letter]}) {domain.capitalize()}'
            letter += 1
            if res is None:
                ax.set_visible(False)
                continue
            keys = ('bias_mean', 'mlp_mean', 'attn_mean') if plot_type == 'states' else \
                   ('bias_mean', 'pre_blank_mean', 'blank_mean')
            sfx = '_signed' if signed else ''
            _draw_bars(ax, *(res[k + sfx] for k in keys), labels, colors, res['num_layer'], 'Layer', '', title)
            _panel(ax, title, res['num_layer'], ('LPD' if signed else Y_LABEL_BARS) if col == 0 else '')
            if signed:
                ax.axhline(res['mean_high_signed'], **REF_CLEAN)
                ax.axhline(res['mean_low_signed'], **REF_CORRUPT)
            ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4, zorder=0)
            if ax.get_legend():
                ax.get_legend().remove()
            if _low_sig(res):
                _mark_low_sig(ax)


def _bars_legend_handles(signed=False):
    from matplotlib.lines import Line2D
    refs = [Line2D([0], [0], label='Clean LPD', **REF_CLEAN),
            Line2D([0], [0], label='Corrupted LPD', **REF_CORRUPT)] if signed else []
    return ([Patch(facecolor=c, label=l) for c, l in zip(STATES_COLORS, STATES_LABELS)] +
            [Patch(facecolor=c, label=l) for c, l in zip(WORDS_COLORS, WORDS_LABELS)] + refs + [_low_sig_handle()])


def save_appendix_lpd_bars(out_dir, family, num_sample=None):
    """A8: LPD counterpart of A1/A2. Rows: Base states, Base words, then Post-trained states and words when the
    family has a post-trained model; columns: domains. Reference lines: clean and corrupted LPD."""
    base, inst, _, i_name = _load_family_pair(family, PAPER_DOMAINS, num_sample)
    rows = [(base, 'states'), (base, 'words')] + ([(inst, 'states'), (inst, 'words')] if i_name else [])
    fig, axes = _paper_fig(len(rows), 3, legend_rows=3)
    _bars_grid(axes, rows, PAPER_DOMAINS, signed=True)
    _share_rows(axes)
    _paper_legend(fig, _bars_legend_handles(signed=True), 3)
    _savepdf(fig, os.path.join(out_dir, f'A8-lpd-bars-{FAMILY_PLOT_DIRS.get(family, family)}.pdf'))


# ── appendix A1: OLMo base + instruct bars (4×3) ─────────────────────────────

def save_appendix_A1_olmo_bars(out_dir, num_sample=None, main_zf=None, family='olmo_1b'):
    """
    Appendix A1: 4×3 grid for OLMo-2-0425-1B (main) and OLMo-2-0425-1B-Instruct (step_2000).
    Row 0 base states · Row 1 base words · Row 2 instruct states · Row 3 instruct words.
    Columns: gender, race, profession.
    States bars use STATES_COLORS; words bars use WORDS_COLORS.
    Single global Y-axis across all 12 panels.
    """
    print(f'  [A1] Loading {family} base + instruct...')
    base_results, inst_results, b_name, i_name = _load_family_pair(
        family, PAPER_DOMAINS, num_sample, main_zf)

    fig, axes = _paper_fig(4, 3, legend_rows=3)
    _bars_grid(axes, [(base_results, 'states'), (base_results, 'words'),
                      (inst_results, 'states'), (inst_results, 'words')], PAPER_DOMAINS)
    _share_rows(axes)
    _paper_legend(fig, _bars_legend_handles(), 3)
    _savepdf(fig, os.path.join(out_dir, 'A1-olmo-bars.pdf' if family == 'olmo_1b'
                               else f'A1-bars-{family}.pdf'))


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

    fig, axes = _paper_fig(2, 3, legend_rows=3)
    _bars_grid(axes, [(pythia_results, 'states'), (pythia_results, 'words')], PAPER_DOMAINS)
    _share_rows(axes)
    _paper_legend(fig, _bars_legend_handles(), 3)
    _savepdf(fig, os.path.join(out_dir, 'A2-pythia-bars.pdf'))


# ── appendix A3: NIE line plots (4×3) ────────────────────────────────────────

_NIE_FIELDS = {'bias': 'bias_mean_signed', 'mlp': 'mlp_mean_signed', 'attn': 'attn_mean_signed',
               'pre_blank': 'pre_blank_mean_signed', 'blank': 'blank_mean_signed',
               'mlp_pre': 'mlp_pre_signed', 'attn_pre': 'attn_pre_signed'}
_NIE_ROWS = {
    'states': (('bias', 'mlp', 'attn'), ['Single state', 'MLP window', 'Attn window'], 'States NIE'),
    'words':  (('bias', 'pre_blank', 'blank'), ['Subject token', 'Pre-target token', 'Target token'], 'Words NIE'),
    'states_pre': (('pre_blank', 'mlp_pre', 'attn_pre'), ['Single state', 'MLP window', 'Attn window'], 'Pre-target NIE'),
}
_LINESTYLES = (STYLE_FIRST, STYLE_SECOND)   # first model solid, second dashed
_RUN_STYLES = (STYLE_PRE, STYLE_POST, STYLE_P2POST, STYLE_POST2PRE)


def _mevery(n, style):
    """Sparse markers, offset per run style so overlapping curves still show distinct markers."""
    step = max(1, n // 8)
    return (_RUN_STYLES.index(style) % step if style in _RUN_STYLES else 0, step)


def _nie_curve(res, key):
    """Signed-chain NIE, as everywhere else; None if missing or the gap is too small to trust."""
    if res is None:
        return None
    arr, gap, low = res.get(_NIE_FIELDS[key]), res.get('effect_gap_signed'), res.get('mean_low_signed')
    if arr is None or low is None or not signed_gap_reliable(gap):
        return None
    return normalized_indirect_effect(arr, low, gap, signed=True)


def _low_sig(res):
    """Low signal = |signed effect gap| (clean - corrupted LPD) below LOW_SIGNAL, as defined in the paper."""
    gap = res.get('effect_gap_signed')
    return (not signed_gap_reliable(gap)) if gap is not None else bool(res.get('low_sig'))


def _low_sig_handle():
    return Patch(facecolor=LOW_SIG_BG, edgecolor=LOW_SIG_COLOR, label='\u26a0 Effect gap < 0.03')


def _mark_low_sig(ax):
    ax.set_facecolor(LOW_SIG_BG)
    ax.text(0.98, 0.97, '\u26a0', transform=ax.transAxes, fontsize=FS_ANNOT, ha='right', va='top',
            color=LOW_SIG_COLOR)


def _nie_line_rows(axes, kinds, models, letter_start=0):
    """NIE line rows on a rows x domains grid. kinds: e.g. ['states', 'words']; models: [(results_by_domain,
    label), ...], drawn with STYLE_FIRST, then STYLE_SECOND. Panels are labelled "(a) Gender"."""
    letter = letter_start
    for row, kind in enumerate(kinds):
        keys, _, _ = _NIE_ROWS[kind]
        colors = BAR_COLORS if kind.startswith('states') else WORDS_COLORS
        for col, domain in enumerate(PAPER_DOMAINS):
            ax, n_lay = axes[row, col], 16
            for j, entry in enumerate(models):
                results, (ls, mk) = entry[0], (entry[2] if len(entry) > 2 else _LINESTYLES[min(j, 1)])
                res = results.get(domain)
                for key, color in zip(keys, colors):
                    nie = _nie_curve(res, key)
                    if nie is not None:
                        n_lay = len(nie)
                        ax.plot(np.arange(len(nie)), nie, color=color, linestyle=ls, marker=mk,
                                markevery=_mevery(len(nie), (ls, mk)), linewidth=2.0, markersize=3.5)
                if res is not None and _low_sig(res):
                    _mark_low_sig(ax)
            ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
            ax.grid(alpha=0.2)
            _panel(ax, f'({"abcdefghijklmnopqrstuvwxyz"[letter]}) {domain.capitalize()}', n_lay,
                   'NIE' if col == 0 else '')
            letter += 1


def _nie_legend_columns(kinds, models):
    """One legend entry per drawn line (condition colour x model line style), grouped into legend columns:
    one column per (kind, model), or one per model when that would give more than four columns."""
    from matplotlib.lines import Line2D

    def entries(kind, j, e):
        _, cond_labels, _ = _NIE_ROWS[kind]
        colors = BAR_COLORS if kind.startswith('states') else WORDS_COLORS
        ls, mk = e[2] if len(e) > 2 else _LINESTYLES[min(j, 1)]
        return [Line2D([0], [0], color=c, linestyle=ls, marker=mk, linewidth=2, markersize=5,
                       label=f'{cond}, {e[1]}') for cond, c in zip(cond_labels, colors)]

    if len(kinds) * len(models) <= 4:
        return [entries(k, j, e) for k in kinds for j, e in enumerate(models)]
    return [[h for k in kinds for h in entries(k, j, e)] for j, e in enumerate(models)]


def _nie_legend(fig, kinds, models=None, blocks=None):
    """Legend below the panels (matplotlib fills a legend column by column)."""
    if blocks:   # models of all blocks, each label once
        models = list({e[1]: e for b in blocks for e in b}.values())
    cols = _nie_legend_columns(kinds, models)
    _paper_legend(fig, [h for c in cols for h in c], len(cols))


def _share_rows(axes):
    """One y-range per row, fitted to that row's data (not forced to start at 0)."""
    for row in np.atleast_2d(axes):
        share_ylim(row)


def _set_ylim(axes, vals):
    if vals:
        m = (max(vals) - min(vals)) * 0.12 or 0.05
        for ax in np.ravel(axes):
            ax.set_ylim(min(vals) - m, max(vals) + m)


def save_appendix_A3_nie_lines(out_dir, num_sample=None, main_zf=None, family='olmo_1b'):
    """A3 (2x3) for one Base/Post family: NIE lines, row 1 states, row 2 words; Base solid, Post dashed."""
    print(f'  [A3] Loading {family} base + instruct...')
    base, inst, _, _ = _load_family_pair(family, PAPER_DOMAINS, num_sample, main_zf)
    models = [(base, LABEL_PRE)] + ([(inst, LABEL_POST)] if inst else [])
    fig, axes = _paper_fig(2, 3, legend_rows=2)
    _nie_line_rows(axes, ['states', 'words'], models)
    _share_rows(axes)
    _nie_legend(fig, ['states', 'words'], models)
    sfx = '' if family == 'olmo_1b' else f'-{family}'
    _savepdf(fig, os.path.join(out_dir, f'A3-nie-lines{sfx}.pdf'))


def save_model_pair_nie(out_dir, fam_a, fam_b, num_sample=None):
    """B: model vs model NIE (columns: domains); model A solid, model B dashed.
    Rows 1-2: Base models (states, words); rows 3-4: Post-trained models, when both families have one."""
    a_base, a_inst, _, a_iname = _load_family_pair(fam_a, PAPER_DOMAINS, num_sample)
    b_base, b_inst, _, b_iname = _load_family_pair(fam_b, PAPER_DOMAINS, num_sample)
    blocks = [[(a_base, _family_label(fam_a)), (b_base, _family_label(fam_b))]]
    if a_iname and b_iname:
        blocks.append([(a_inst, _family_label(fam_a)), (b_inst, _family_label(fam_b))])
    fig, axes = _paper_fig(2 * len(blocks), 3, legend_rows=5)
    for i, models in enumerate(blocks):
        _nie_line_rows(axes[2 * i:2 * i + 2], ['states', 'words'], models, letter_start=6 * i)
    _share_rows(axes)
    _nie_legend(fig, ['states', 'words'], blocks=blocks)
    name = lambda f: FAMILY_PLOT_DIRS.get(f, f)
    _savepdf(fig, os.path.join(out_dir, f'B-nie-{name(fam_a)}-vs-{name(fam_b)}.pdf'))


# ── main body: one row of panels, one model family per panel ─────────────────

MAIN_FAMILIES = ['olmo_1b', 'qwen2.5_1.5b', 'llama3.2_1b', 'gemma3_1b', 'pythia']


def _family_label(family):
    return 'Pythia 1B' if family == 'pythia' else CROSS_PATCH_FAMILIES[family]['label']


def save_main_figures(out_dir, domain='gender', num_sample=None):
    """
    Main-body figures, one panel per model family (each panel keeps its own layer axis):
      main-nie-prepost-<domain>.pdf   1x5  NIE at subject / pre-target / target tokens, Base solid vs Post-trained
                                           dashed (Pythia: Base only), shared legend
      main-alp-prepost-<domain>.pdf   1x4  single-state ALP bars at the subject token, Base vs Post-trained
    """
    from matplotlib.lines import Line2D
    pairs = {f: _load_family_pair(f, [domain], num_sample) for f in MAIN_FAMILIES}
    def nie_panel(ax, k, n_cols, family, styled, keys, colors):
        res = None
        for results, (ls, mk) in styled:
            res = results.get(domain)
            for key, color in zip(keys, colors):
                nie = _nie_curve(res, key)
                if nie is not None:
                    ax.plot(np.arange(len(nie)), nie, color=color, linestyle=ls, marker=mk,
                            markevery=_mevery(len(nie), (ls, mk)), linewidth=2.0, markersize=3.5)
            if res is not None and _low_sig(res):
                _mark_low_sig(ax)
        ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        ax.grid(alpha=0.2)
        _panel(ax, _family_label(family), res['num_layer'], 'NIE' if k == 0 else '', n_cols)

    n = len(MAIN_FAMILIES)
    for kind, colors, name in [('words', WORDS_COLORS, 'main-nie-prepost'), ('states', BAR_COLORS, 'main-nie-states-prepost'),
                               ('states_pre', BAR_COLORS, 'main-nie-states-pretarget-prepost')]:
        fig, axes = _paper_fig(1, n, legend_rows=2, row_h=PAPER_ROW_H)
        for k, fam in enumerate(MAIN_FAMILIES):
            styled = [(pairs[fam][0], STYLE_FIRST)] + ([(pairs[fam][1], STYLE_SECOND)] if pairs[fam][3] else [])
            nie_panel(axes[0, k], k, n, fam, styled, _NIE_ROWS[kind][0], colors)
            if k:
                axes[0, k].tick_params(labelleft=False)   # shared y-range: ticks labelled on the first panel only
        share_ylim(axes)
        _nie_legend(fig, [kind], [(None, LABEL_PRE), (None, LABEL_POST)])
        fig.subplots_adjust(wspace=0.06)   # narrow gaps between panels, so each x-axis is longer
        _savepdf(fig, os.path.join(out_dir, f'{name}-{domain}.pdf'))

    post = [f for f in MAIN_FAMILIES if pairs[f][3]]
    fig, axes = _paper_fig(1, len(post), legend_rows=1, row_h=PAPER_ROW_H)
    width = 0.4
    for k, fam in enumerate(post):
        b, i = (pairs[fam][j].get(domain) for j in (0, 1))
        ax, xs = axes[0, k], np.arange(b['num_layer'])
        ax.bar(xs - width / 2, b['bias_mean'], width, color=BASE_COLOR, linewidth=0)
        ax.bar(xs + width / 2, i['bias_mean'], width, color=INSTRUCT_COLOR, linewidth=0)
        ax.grid(alpha=0.2)
        _panel(ax, _family_label(fam), b['num_layer'], Y_LABEL_BARS if k == 0 else '', len(post))
        if _low_sig(b) or _low_sig(i):
            _mark_low_sig(ax)
        lo = min(b['bias_mean'].min(), i['bias_mean'].min())
        hi = max(b['bias_mean'].max(), i['bias_mean'].max())
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo))
    _paper_legend(fig, [Patch(facecolor=BASE_COLOR, label=LABEL_PRE),
                        Patch(facecolor=INSTRUCT_COLOR, label=LABEL_POST), _low_sig_handle()], 3)
    _savepdf(fig, os.path.join(out_dir, f'main-alp-prepost-{domain}.pdf'))


def save_main_crosspatch(out_dir, domain='gender', num_sample=None):
    """main-crosspatch-<domain>.pdf (1x4): single-state subject NIE, within-model (solid) vs cross-model (dashed),
    for the patched Post model (Pre->Post) and the patched Pre model (Post->Pre)."""
    from matplotlib.lines import Line2D
    fams = [f for f in MAIN_FAMILIES if f != 'pythia']
    fig, axes = _paper_fig(1, len(fams), legend_rows=2, row_h=PAPER_ROW_H)
    for k, fam in enumerate(fams):
        pre, post, _, _ = _load_family_pair(fam, [domain], num_sample)
        lines = [(post, INSTRUCT_COLOR, STYLE_FIRST),
                 ({domain: load_cross_patch_domain('pre_to_post', domain, num_sample, family=fam)}, INSTRUCT_COLOR, STYLE_SECOND),
                 (pre, BASE_COLOR, STYLE_FIRST),
                 ({domain: load_cross_patch_domain('post_to_pre', domain, num_sample, family=fam)}, BASE_COLOR, STYLE_SECOND)]
        ax = axes[0, k]
        for results, color, (ls, mk) in lines:
            nie = _nie_curve(results.get(domain), 'bias')
            if nie is not None:
                ax.plot(np.arange(len(nie)), nie, color=color, linestyle=ls, marker=mk, linewidth=2.0, markersize=3)
        ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        ax.grid(alpha=0.2)
        _panel(ax, _family_label(fam), pre[domain]['num_layer'], 'NIE' if k == 0 else '', len(fams))
        if k:
            ax.tick_params(labelleft=False)
    share_ylim(axes)
    handles = [Line2D([0], [0], color=c, linestyle=ls, marker=mk, linewidth=2, label=lab) for c, (ls, mk), lab in [
        (INSTRUCT_COLOR, STYLE_FIRST, 'Post patched, within-model'),
        (INSTRUCT_COLOR, STYLE_SECOND, 'Post patched with Pre (Pre$\\to$Post)'),
        (BASE_COLOR, STYLE_FIRST, 'Pre patched, within-model'),
        (BASE_COLOR, STYLE_SECOND, 'Pre patched with Post (Post$\\to$Pre)')]]
    _paper_legend(fig, handles, 2)
    fig.subplots_adjust(wspace=0.06)
    _savepdf(fig, os.path.join(out_dir, f'main-crosspatch-{domain}.pdf'))


def save_main_compact(out_dir, domain='gender', num_sample=None):
    """Single-column main figures:
      main-nie-prepost-3models-<domain>.pdf  2x2, OLMo 2 / Qwen2.5 / Pythia token NIE, Pre solid vs Post dashed,
                                             legend in the fourth cell
      main-crosspatch-tokens[-notarget]-<domain>.pdf  1x2, OLMo 2 / Qwen2.5 cross-model token NIE,
                                             Pre->Post solid vs Post->Pre dashed"""
    from matplotlib.lines import Line2D
    keys, labels, _ = _NIE_ROWS['words']

    def lines(ax, styled, ks):
        for results, (ls, mk) in styled:
            for key, color in zip(keys, WORDS_COLORS):
                nie = _nie_curve(results.get(domain), key) if key in ks else None
                if nie is not None:
                    ax.plot(np.arange(len(nie)), nie, color=color, linestyle=ls, marker=mk,
                            markevery=_mevery(len(nie), (ls, mk)), linewidth=2.0, markersize=3.5)
        ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        ax.grid(alpha=0.2)

    def handles(ks, styles):
        """One legend entry per drawn line: token colour x run style."""
        return [Line2D([0], [0], color=c, linestyle=ls, marker=mk, linewidth=2, label=f'{lab}, {name}')
                for name, (ls, mk) in styles for k, lab, c in zip(keys, labels, WORDS_COLORS) if k in ks]

    fams = ['olmo_1b', 'qwen2.5_1.5b', 'pythia']
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_COL_WIDTH, 2 * PAPER_COL_WIDTH / 2 * 0.72))
    for ax, fam in zip(axes.flat, fams):
        pre, post, _, inst = _load_family_pair(fam, [domain], num_sample)
        lines(ax, [(pre, STYLE_FIRST)] + ([(post, STYLE_SECOND)] if inst else []), keys)
        _panel(ax, _family_label(fam), pre[domain]['num_layer'], 'NIE', 4)
    share_ylim(axes[:, :].flat[:3])
    axes[1, 1].axis('off')
    axes[1, 1].legend(handles=handles(keys, [('Pre-trained model (Pre)', STYLE_PRE),
                                             ('Post-trained model (Post)', STYLE_POST)]),
                      loc='center left', bbox_to_anchor=(-0.08, 0.5), fontsize=PAPER_FS_LEGEND - 2, frameon=True,
                      handlelength=2.2)
    fig.tight_layout()
    _savepdf(fig, os.path.join(out_dir, f'main-nie-prepost-3models-{domain}.pdf'))

    for ks, sfx in [(keys, ''), (keys[:2], '-notarget')]:
        fig, axes = _paper_fig(1, 2, legend_rows=len(ks) * 2, width=PAPER_COL_WIDTH, row_h=2.3)
        for ax, fam in zip(axes[0], ['olmo_1b', 'qwen2.5_1.5b']):
            pre, post, _, _ = _load_family_pair(fam, [domain], num_sample)
            p2post = {domain: load_cross_patch_domain('pre_to_post', domain, num_sample, family=fam)}
            p2pre = {domain: load_cross_patch_domain('post_to_pre', domain, num_sample, family=fam)}
            lines(ax, [(pre, STYLE_PRE), (post, STYLE_POST), (p2post, STYLE_P2POST), (p2pre, STYLE_POST2PRE)], ks)
            _panel(ax, _family_label(fam), p2post[domain]['num_layer'], 'NIE', 4)
        share_ylim(axes)
        _paper_legend(fig, handles(ks, [(LABEL_PRE, STYLE_PRE), (LABEL_POST, STYLE_POST),
                                        (LABEL_P2POST, STYLE_P2POST), (LABEL_POST2PRE, STYLE_POST2PRE)]), 2)
        _savepdf(fig, os.path.join(out_dir, f'main-crosspatch-tokens{sfx}-{domain}.pdf'))


def save_main_domain_compare(out_dir, num_sample=None):
    """
    main-domains.pdf (single column): clean LPD per domain for all nine models, grouped by model, with 95%
    bootstrap intervals from tables/T6_bias_types.csv when scripts/tables.py has been run.
    """
    domains = ['gender', 'profession', 'race']
    models, groups = [], []   # (tick label, results) per bar group; (family label, first index, last index)
    for fam in MAIN_FAMILIES:
        base, inst, _, i_name = _load_family_pair(fam, domains, num_sample)
        start = len(models)
        models.append(('Pre', base))
        if i_name:
            models.append(('Post', inst))
        groups.append((_family_label(fam).rsplit(' ', 1)[0], start, len(models) - 1))
    ci_path = os.path.join(TABLES_DIR, 'T6_bias_types.csv')
    ci = None
    if os.path.exists(ci_path):
        import pandas as pd
        ci = pd.read_csv(ci_path)
    fig, axes = _paper_fig(1, 1, legend_rows=1, width=PAPER_COL_WIDTH)
    ax, width = axes[0, 0], 0.27
    xs = np.arange(len(models))
    for k, d in enumerate(domains):
        vals = [res[d]['mean_high_signed'] for _, res in models]
        err = None
        if ci is not None and len(ci) == len(models):
            lo, hi = ci[f'Delta_clean_{d}_lo'].to_numpy(), ci[f'Delta_clean_{d}_hi'].to_numpy()
            err = [np.array(vals) - lo, hi - np.array(vals)]
        ax.bar(xs + (k - 1) * width, vals, width, color=DOMAIN_COLORS[d], linewidth=0, label=d.capitalize(),
               yerr=err, error_kw=dict(elinewidth=0.8, capsize=1.5, ecolor='black'))
    ax.set_xticks(xs)
    ax.set_xticklabels([m for m, _ in models], fontsize=PAPER_FS_TICK - 3)
    for name, first, last in groups:
        ax.text((first + last) / 2, -0.17, name, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=PAPER_FS_TICK - 2)
    ax.set_ylabel('LPD', fontsize=PAPER_FS_LABEL)
    ax.tick_params(axis='y', labelsize=PAPER_FS_TICK)
    ax.grid(axis='y', alpha=0.2)
    handles, _ = ax.get_legend_handles_labels()
    _paper_legend(fig, handles, 3)
    _savepdf(fig, os.path.join(out_dir, 'main-domains.pdf'))


def save_crosspatch_nie_overlay(out_dir, num_sample=None, main_zf=None, family='olmo_1b'):
    """
    A7 (4x3): within-model vs cross-model patching of the same model, NIE (line styles from plot_utils).
      rows 1-2: Pre -> Post (the Post model is patched), states and words
      rows 3-4: Post -> Pre (the Pre model is patched), states and words
    Both use the patched model's own clean and corrupted passes, so they are directly comparable.
    """
    print(f'  [A7] Loading {family} Pre, Post + cross-patch...')
    pre, post, _, _ = _load_family_pair(family, PAPER_DOMAINS, num_sample, main_zf)
    p2post = {d: load_cross_patch_domain('pre_to_post', d, num_sample, family=family) for d in PAPER_DOMAINS}
    p2pre = {d: load_cross_patch_domain('post_to_pre', d, num_sample, family=family) for d in PAPER_DOMAINS}
    blocks = [[(post, LABEL_POST, STYLE_POST), (p2post, LABEL_P2POST, STYLE_P2POST)],
              [(pre, LABEL_PRE, STYLE_PRE), (p2pre, LABEL_POST2PRE, STYLE_POST2PRE)]]
    fig, axes = _paper_fig(4, 3, legend_rows=6)
    for i, models in enumerate(blocks):
        _nie_line_rows(axes[2 * i:2 * i + 2], ['states', 'words'], models, letter_start=6 * i)
    for r, ax in enumerate(axes.flat):
        ax.set_title(ax.get_title() + (': Post vs. Pre$\\to$Post' if r < 6 else ': Pre vs. Post$\\to$Pre'), fontsize=PAPER_FS_TITLE)
    _share_rows(axes)
    _nie_legend(fig, ['states', 'words'], blocks=blocks)
    _savepdf(fig, os.path.join(out_dir, 'A7-crosspatch-nie-overlay.pdf' if family == 'olmo_1b'
                               else f'A7-crosspatch-nie-overlay-{family}.pdf'))


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

    fig, axes = _paper_fig(4, 3, legend_rows=3)
    _bars_grid(axes, [(pre, 'states'), (pre, 'words'), (post, 'states'), (post, 'words')], PAPER_DOMAINS)
    _share_rows(axes)
    _paper_legend(fig, _bars_legend_handles(), 3)
    _savepdf(fig, os.path.join(out_dir, out_name))


# ── appendix A5: effect gap trajectory (3×1) ─────────────────────────────────

def save_appendix_A5_trajectory(base_stats, instruct_stats, out_dir, out_name='A5-trajectory.pdf'):
    """
    Appendix A5: 3×1 effect gap trajectory for OLMo base + instruct.
    One panel per paper domain (rows); panel title = domain name only.
    OLMo base (blue solid) and instruct (orange dashed) plotted on the same x-axis.
    Y-axis shared across all 3 panels.
    """
    letters = ['a', 'b', 'c']

    domain_data = {}
    for domain in PAPER_DOMAINS:
        base_pts, instruct_pts = [], []
        for e in base_stats:
            s = e['domains'].get(domain)
            if s:
                base_pts.append((e['label'], s['effect_gap_signed'], abs(s['effect_gap_signed']) < LOW_SIGNAL))
        for e in instruct_stats:
            s = e['domains'].get(domain)
            if s:
                instruct_pts.append((e['label'], s['effect_gap_signed'], abs(s['effect_gap_signed']) < LOW_SIGNAL))
        if base_pts or instruct_pts:
            domain_data[domain] = (base_pts, instruct_pts)

    if not domain_data:
        print('  [A5] No data; skipping.')
        return

    all_gaps = [p[1] for bp, ip in domain_data.values() for p in bp + ip]
    if not all_gaps:
        return
    margin = (max(all_gaps) - min(all_gaps)) * 0.12 or 0.005
    y_min  = min(all_gaps) - margin
    y_max  = max(all_gaps) + margin

    fig, axes = _paper_fig(3, 1, legend_rows=1, row_h=PAPER_TRAJ_ROW_H)
    axes = axes[:, 0]

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
                color=BASE_COLOR, label=LABEL_PRE,
                linewidth=2, markersize=7, zorder=3)
        if instruct_pts:
            bridge_xs = [n_base - 1] + list(xs[n_base:])
            bridge_ys = [gaps[n_base - 1]] + list(gaps[n_base:])
            ax.plot(bridge_xs, bridge_ys, 's--',
                    color=INSTRUCT_COLOR, label=LABEL_POST,
                    linewidth=2, markersize=7, zorder=3)
            ax.axvline(n_base - 0.5, color='gray', linestyle='--',
                       linewidth=1.2, alpha=0.6)

        for xi, ls in enumerate(low_sig):
            if ls:
                ax.annotate('⚠', (xs[xi], gaps[xi]),
                            textcoords='offset points', xytext=(0, 6),
                            ha='center', fontsize=PAPER_FS_LABEL, color=LOW_SIG_COLOR)

        ax.set_xticks(xs)
        ax.set_xticklabels([CKPT_TICKS.get(l, l) for l in labels_x], fontsize=PAPER_FS_TICK)
        ax.set_xlabel('Training checkpoint', fontsize=PAPER_FS_LABEL)
        ax.set_ylabel('Effect gap', fontsize=PAPER_FS_LABEL)
        ax.set_title(f'({letter}) {domain.capitalize()}', fontsize=PAPER_FS_TITLE)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='y', labelsize=PAPER_FS_TICK)
        ax.grid(axis='y', alpha=0.25)
        ax.axhline(0, color='black', linewidth=0.7, alpha=0.3)

    handles, _ = axes[0].get_legend_handles_labels()
    _paper_legend(fig, handles + [_low_sig_handle()], 3)
    _savepdf(fig, os.path.join(out_dir, out_name))


# ── appendix A6: NIE heatmap (layer × checkpoint) ────────────────────────────

_A6_CMAP = plt.get_cmap('RdBu_r').copy()
_A6_CMAP.set_bad(LOW_SIG_BG)


def save_appendix_A6_heatmap(base_stats, instruct_stats, out_dir, num_sample=None,
                             out_name='A6-heatmap.pdf', step2000_model='OLMo-2-0425-1B-Instruct'):
    """
    Appendix A6: 3×3 grid of NIE heatmaps.
    Rows: states conditions (full restore, MLP-window restore, Attn-window restore).
    Columns: domains (gender, race, profession).
    X-axis: training checkpoints (base pre-training | instruct fine-tuning).
    Y-axis: layer (0–15).
    step_2000 is missing from instruct stats.json NIE arrays; loaded from local NFS.
    Colorscale shared per row (same condition, all 3 domains).
    """
    # (display label, stats.json key, load_within_model_from_local dict key)
    # attn_score in stats = _attn.npz = restored window of Attn outputs
    # mlp_score  in stats = _mlp.npz  = restored window of MLP outputs
    conditions = [
        (STATES_LABELS[0], 'states_score_signed', 'bias_mean_signed'),
        (STATES_LABELS[1], 'mlp_score_signed',    'mlp_mean_signed'),
        (STATES_LABELS[2], 'attn_score_signed',   'attn_mean_signed'),
    ]

    # Load step_2000 from local NFS (full result dict, signed chain included via
    # _finalize_result_dict). Other checkpoints are read from stats.json, which stores
    # the same signed chain (see the per-checkpoint loop that builds all_ckpt_stats) --
    # both go through _to_nie below, same as A3.
    step2000_local = {}
    for domain in (PAPER_DOMAINS if step2000_model else []):
        res = load_within_model_from_local(
            step2000_model, 'allenai', 'step_2000', domain, num_sample)
        if res is not None:
            step2000_local[domain] = res

    def _to_nie(signed_arr, mean_low_signed, effect_gap_signed):
        """Signed-chain NIE; None if the array, low anchor, or gap magnitude
        (see signed_gap_reliable) aren't available."""
        if signed_arr is None or mean_low_signed is None or not signed_gap_reliable(effect_gap_signed):
            return None
        return normalized_indirect_effect(signed_arr, mean_low_signed, effect_gap_signed,
                                          degenerate='zero', signed=True)

    def _get_from_stats(stats_list, domain, key):
        """Return [(label, nie_array)] from the domain's signed chain. Points with no
        signed data, or too small a signed gap to trust, are omitted."""
        out = []
        for e in stats_list:
            d = e['domains'].get(domain, {})
            nie = _to_nie(d.get(key), d.get('mean_low_signed'), d.get('effect_gap_signed'))
            if nie is not None:
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
                    nie = _to_nie(r.get(local_key), r.get('mean_low_signed'), r.get('effect_gap_signed'))
                    if nie is not None:
                        inst_ordered.append(('step2000', nie))
                step2000_inserted = True
        if not step2000_inserted and domain in step2000_local:
            r = step2000_local[domain]
            nie = _to_nie(r.get(local_key), r.get('mean_low_signed'), r.get('effect_gap_signed'))
            if nie is not None:
                inst_ordered.append(('step2000', nie))

        all_pts = base_pts + inst_ordered
        if not all_pts:
            return None, None, 0

        labels_x = [p[0] for p in all_pts]
        matrix   = np.stack([p[1] for p in all_pts], axis=1)  # (num_layer, n_checkpoints)
        return matrix, labels_x, len(base_pts)

    fig, axes = _paper_fig(3, 3, legend_rows=1)
    fig.set_figheight(fig.get_figheight() + 3 * 1.0)   # room for vertical checkpoint ticks

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
                cmap=_A6_CMAP, vmin=vmin, vmax=vmax,
                extent=[-0.5, matrix.shape[1] - 0.5, -0.5, num_layer - 0.5])

            # Vertical separator between base pre-training and instruct fine-tuning
            if n_base_ckpts < matrix.shape[1]:
                ax.axvline(n_base_ckpts - 0.5, color='black', linewidth=1.5,
                           linestyle='--', alpha=0.7)

            ax.set_xticks(np.arange(len(labels_x)))
            ax.set_xticklabels([CKPT_TICKS.get(l, l).replace('\n', ' ') for l in labels_x], rotation=90,
                               fontsize=PAPER_FS_TICK - 2)
            ax.set_yticks(np.arange(0, num_layer, 2))
            ax.tick_params(axis='y', labelsize=PAPER_FS_TICK)

            if row_i == 0:
                ax.set_title(f'({"abc"[col_i]}) {domain.capitalize()}', fontsize=PAPER_FS_TITLE)

            if col_i == 0:
                ax.set_ylabel('Layer', fontsize=PAPER_FS_LABEL)
                ax.annotate(cond_label, xy=(-0.28, 0.5), xycoords='axes fraction',
                            rotation=90, ha='center', va='center',
                            fontsize=PAPER_FS_LABEL)
            else:
                ax.set_ylabel('')

            if row_i == len(conditions) - 1:
                ax.set_xlabel('Checkpoint', fontsize=PAPER_FS_LABEL)

            cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
            cb.ax.tick_params(labelsize=PAPER_FS_TICK - 2)
            if col_i == 2:
                cb.set_label('NIE', fontsize=PAPER_FS_LABEL)
            else:
                cb.set_label('')

    _paper_legend(fig, [Patch(facecolor=LOW_SIG_BG, edgecolor=LOW_SIG_COLOR,
                              label='\u26a0 Effect gap < 0.03 (checkpoint not shown)')], 1)
    _savepdf(fig, os.path.join(out_dir, out_name))


# ── main loop ─────────────────────────────────────────────────────────────────

print(f'Opening zip: {ZIP_PATH}')
zf = zipfile.ZipFile(ZIP_PATH, 'r')
zip_names_all = zf.namelist()

all_models_stats = {}  # accumulate for cross-model plots

for model_name in (models_to_run if RUN_BARS or RUN_DELTA or RUN_COMPARE else []):
    cfg = MODEL_CONFIGS[model_name]
    org = cfg['org']

    model_out_dir  = _out(os.path.join(PLOTS_BASE, model_name),
                          'pythia' if model_name.startswith('pythia') else 'olmo_1b',
                          'ALP', os.path.join('checkpoints', model_name))
    all_ckpt_stats = []

    # cross-checkpoint accumulator: domain → list of (label, r1, r2, r3, nl)
    cross_states = {d: [] for d in BIAS_TYPES}
    cross_words  = {d: [] for d in BIAS_TYPES}

    for checkpoint, ckpt_label in cfg['checkpoints']:
        print(f'\n=== {model_name}  [{ckpt_label}] ===')
        out_ckpt_dir = os.path.join(model_out_dir, ckpt_label)

        states_data = {}
        words_data  = {}
        ckpt_stats  = {'checkpoint': checkpoint, 'label': ckpt_label, 'domains': {}}

        for domain in domains_to_run:
            print(f'  {domain}')

            local_dir  = local_cases_dir(model_name, org, checkpoint, domain)
            zip_prefix = zip_cases_prefix(org, model_name, checkpoint, domain)
            zip_has_data = any(n.startswith(zip_prefix) and n.endswith('.npz')
                               for n in zip_names_all)
            use_local  = (args.source == 'local') or USE_PKG_RESULTS or \
                         (args.source == 'auto' and os.path.isdir(local_dir)
                          and not zip_has_data)
            if use_local and not os.path.isdir(local_dir):
                print(f'    No local data: {local_dir}; skipping.')
                continue

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

            # Signed chain, for NIE only (ALP above stays on the abs chain). None when the
            # result files predate scores_signed -- report.md/stats.json fall back to "n/a"
            # for NIE rather than silently mixing signed and abs numbers.
            bias_mean_s, _, _, _, mean_high_s, mean_low_s = \
                collect_scores(single_items, loader, signed=True)
            attn_mean_s, _, _, _, _, _ = collect_scores(attn_items, loader, signed=True)
            mlp_mean_s,  _, _, _, _, _ = collect_scores(mlp_items,  loader, signed=True)

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
                # abs chain -- ALP bars, peak-layer bookkeeping
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
                # signed chain -- NIE only. None when scores_signed isn't in the result files.
                # Gated on bias_mean_s (the array collect_scores actually sets to None on
                # failure) rather than the scalar mean_high_s/mean_low_s: collect_scores's
                # failure tuple is (None, None, None, 0, 0.0, 0.0) -- the scalars come back
                # as 0.0, not None, so checking "mean_high_s is not None" alone would store
                # effect_gap_signed=0.0 (truthy for a None-check) while states_score_signed
                # stayed None, and every reader below would then pass its own is-not-None
                # guard and crash indexing into None.
                'mean_high_signed':  round(mean_high_s, 6) if bias_mean_s is not None else None,
                'mean_low_signed':   round(mean_low_s,  6) if bias_mean_s is not None else None,
                'effect_gap_signed': (round(mean_high_s - mean_low_s, 6)
                                      if bias_mean_s is not None else None),
                'states_score_signed': bias_mean_s.tolist() if bias_mean_s is not None else None,
                'attn_score_signed':   attn_mean_s.tolist() if attn_mean_s is not None else None,
                'mlp_score_signed':    mlp_mean_s.tolist()  if mlp_mean_s  is not None else None,
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
    compare_dir = _out(os.path.join(PLOTS_BASE, 'compare'), 'olmo_1b', 'ALP',
                       os.path.join('checkpoints', 'compare'))
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
    USE_PKG_RESULTS or os.makedirs(cross_patch_out, exist_ok=True)

    all_direction_results = {}  # olmo_1b results — {direction_key: {domain: result_dict}}
    fam_results_all = {}        # {family: {direction_key: {domain: result_dict}}}

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

            dir_out = _out(os.path.join(fam_out, direction_key), cp_family, 'ALP',
                           'prepost' if direction_key == 'pre_to_post' else 'postpre')

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
            save_cross_patch_comparison(fam_direction_results, domains_to_run,
                                        _out(fam_out, cp_family, 'ALP', 'cross'),
                                        family_label=title_label)
        elif len(fam_direction_results) == 1:
            print('\n  Only one direction has data — skipping comparison plots.')

        if is_olmo:
            all_direction_results = fam_direction_results
        fam_results_all[cp_family] = fam_direction_results

    # 4-panel comparison: within-model last checkpoints + both cross-patch directions
    # Base: OLMo-2-0425-1B main checkpoint (from main.zip)
    # Instruct: OLMo-2-0425-1B-Instruct step_2000 main checkpoint (from local NFS)
    for cp_family, fam_dir_results in fam_results_all.items():
        if not fam_dir_results:
            continue
        is_olmo = cp_family == 'olmo_1b'
        print(f'\n  Loading within-model base + instruct for 4-panel plots ({cp_family})...')
        _b, _i, b_name, i_name = _load_family_pair(cp_family, domains_to_run, args.num_sample)
        within_model_panels = {d: {'base_last': _b[d], 'inst_last': _i[d]} for d in domains_to_run}
        legacy = cross_patch_out if is_olmo else os.path.join(cross_patch_out, cp_family)
        save_cross_patch_4panel(within_model_panels, fam_dir_results, domains_to_run,
                                _out(legacy, cp_family, 'ALP', 'cross'),
                                inst_label=f'{i_name}\n(post)', base_label=f'{b_name}\n(pre)',
                                file_prefix='4panel-step2000' if is_olmo else f'4panel-{cp_family}',
                                extra_out_dir=None if USE_PKG_RESULTS else os.path.join(PLOTS_BASE, 'main_body'),
                                within_out_dir=_out(legacy, cp_family, 'ALP', 'within'))
        if USE_PKG_RESULTS and 'gender' in domains_to_run:
            import shutil
            _f = ('4panel-step2000' if is_olmo else f'4panel-{cp_family}') + '-gender-states.pdf'
            shutil.copy(os.path.join(_out(legacy, cp_family, 'ALP', 'cross'), _f),
                        _paper(legacy, None if is_olmo else cp_family))

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
                # signed chain, needed by the A6 NIE heatmap
                **{k: (None if res.get(v) is None else np.asarray(res[v]).tolist())
                   for k, v in (('states_score_signed', 'bias_mean_signed'),
                                ('mlp_score_signed', 'mlp_mean_signed'),
                                ('attn_score_signed', 'attn_mean_signed'))},
                'mean_low_signed':   res.get('mean_low_signed'),
                'effect_gap_signed': res.get('effect_gap_signed'),
            }
        entries.append(entry)
    n_loaded = sum(len(e['domains']) for e in entries)
    print(f'  [NPZ] {model_name}: {n_loaded} domain entries across {len(entries)} checkpoints')
    return entries


# ── appendix figures ──────────────────────────────────────────────────────────
if RUN_APPENDIX:
    print('\n=== Appendix figures ===')
    appendix_out = os.path.join(PLOTS_BASE, 'appendix')
    USE_PKG_RESULTS or os.makedirs(appendix_out, exist_ok=True)

    _app_main_zf = zipfile.ZipFile(MAIN_ZIP, 'r')

    for _fam in cp_families_to_run:
        print(f'\n  A1: {_fam} base + instruct bar charts (4×3)...')
        save_appendix_A1_olmo_bars(_paper(appendix_out, _fam), args.num_sample,
                                   main_zf=_app_main_zf, family=_fam)

    print('\n  A2: Pythia bar charts (2×3)...')
    save_appendix_A2_pythia_bars(_paper(appendix_out, 'pythia'), args.num_sample)
    for _fam in list(CROSS_PATCH_FAMILIES) + ['pythia']:
        save_appendix_lpd_bars(_paper(appendix_out, _fam), _fam, args.num_sample)

    for _fam in list(cp_families_to_run) + ['pythia']:
        print(f'\n  A3-1 / A3-2: {_fam}...')
        save_appendix_A3_nie_lines(_paper(appendix_out, _fam), args.num_sample,
                                   main_zf=_app_main_zf, family=_fam)

    print('\n  B: model vs model NIE, every pair...')
    _all_models = list(cp_families_to_run) + ['pythia']
    for _i, _fa in enumerate(_all_models):
        for _fb in _all_models[_i + 1:]:
            save_model_pair_nie(_paper(appendix_out, 'model_comparison'), _fa, _fb, args.num_sample)

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
        save_appendix_A4_cross_patch_bars(_app_cp, _paper(appendix_out, _fam), out_name=_fname)

    print('\n  A5: Trajectory plots (1×3)...')
    _base_traj     = _load_checkpoints_from_npz('OLMo-2-0425-1B',          args.num_sample)
    _instruct_traj = _load_checkpoints_from_npz('OLMo-2-0425-1B-Instruct', args.num_sample)
    _all_ckpts = lambda t: t and all(set(PAPER_DOMAINS) <= set(e['domains']) for e in t)
    if not (_all_ckpts(_base_traj) and _all_ckpts(_instruct_traj)):
        print('  [A5/A6] Some checkpoints are not finished yet; skipping both.')
        _base_traj = _instruct_traj = None
    if _base_traj and _instruct_traj:
        save_appendix_A5_trajectory(_base_traj, _instruct_traj, _paper(appendix_out, 'olmo_1b'))
    else:
        print('  [A5] Skipping.')

    print('\n  Pythia: effect gap over its checkpoints...')
    _pythia_traj = _load_checkpoints_from_npz('pythia-1b', args.num_sample)
    if _all_ckpts(_pythia_traj):
        save_appendix_A5_trajectory(_pythia_traj, [], _paper(appendix_out, 'pythia'),
                                    out_name='A5-trajectory-pythia.pdf')
    else:
        print('  [Pythia A5] Some checkpoints are not finished yet; skipping.')

# ── cross-patch NIE overlay (within-model vs cross-patch, 4×3) ────────────────
if RUN_CP_NIE:
    print('\n=== Cross-patch NIE overlay (within-model vs cross-patch) ===')
    cp_nie_out = os.path.join(PLOTS_BASE, 'appendix')
    USE_PKG_RESULTS or os.makedirs(cp_nie_out, exist_ok=True)
    cp_mb_out = os.path.join(PLOTS_BASE, 'main_body')
    USE_PKG_RESULTS or os.makedirs(cp_mb_out, exist_ok=True)
    for _fam in cp_families_to_run:
        save_crosspatch_nie_overlay(_paper(cp_nie_out, _fam), args.num_sample, family=_fam)

if RUN_MAIN:
    print('\n=== Main-body figures ===')
    save_main_figures(_paper(PLOTS_BASE), num_sample=args.num_sample)
    save_main_domain_compare(_paper(PLOTS_BASE), num_sample=args.num_sample)
    save_main_crosspatch(_paper(PLOTS_BASE), num_sample=args.num_sample)
    save_main_compact(_paper(PLOTS_BASE), num_sample=args.num_sample)

zf.close()
print('\nDone.')
