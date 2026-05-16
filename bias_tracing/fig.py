"""
Generate all bias-tracing bar-chart plots.

──────────────────────────────────────────────────────────────────────────────
WITHIN-MODEL PATCHING  (--plots bars | delta | compare)
──────────────────────────────────────────────────────────────────────────────
Standard causal tracing: corrupt the subject tokens of one model, restore one
hidden state at a time, measure how much the bias-consistent prediction recovers.

Per checkpoint  →  plots/{model}/{label}/
    {domain}-states.pdf         full / Attn-severed / MLP-severed bars per layer
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

Per direction  →  plots/cross_patch/{direction}/
    {domain}-states.pdf         bars per layer (same 3-bar layout as within-model)
    {domain}-words.pdf
    composite-states.pdf        all 4 domains side-by-side
    composite-words.pdf
    composite-all.pdf           2×4 grid

Comparison (both directions)  →  plots/cross_patch/
    {domain}-directions-states.pdf   pre→post vs post→pre, fixed Y-axis
    {domain}-directions-words.pdf

Data source: local filesystem at CROSS_PATCH_BASE (defined in plot_utils.py).
Use --direction to run only one direction.
"""

import os
import io
import json
import zipfile
import argparse
import datetime
import math
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_utils import (
    ZIP_PATH, LOCAL_BASE, PLOTS_BASE,
    MODEL_CONFIGS, BIAS_TYPES,
    CROSS_PATCH_BASE, CROSS_PATCH_CONFIGS,
    STATES_LABELS, WORDS_LABELS, BAR_COLORS, LOW_SIGNAL, Y_LABEL_BARS,
    FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    FIG_BAR_W_SINGLE, FIG_BAR_H_SINGLE, FIG_BAR_W_PER_COL,
    FIG_GRID_W_PER_COL, FIG_ROW_H, FIG_LINE_W_PER_PAN, FIG_TRAJ_H,
    BASE_COLOR, INSTRUCT_COLOR, LOW_SIG_COLOR, LOW_SIG_BG,
    local_cases_dir, zip_cases_prefix, partition_names,
    load_npz_local, load_npz_zip,
    collect_scores, _draw_bars, _savepdf,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

PLOT_CHOICES = ['bars', 'delta', 'compare', 'cross_patch']

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
args = parser.parse_args()

_plots = set(args.plots)
if 'all' in _plots:
    _plots = set(PLOT_CHOICES)
RUN_BARS        = 'bars'        in _plots
RUN_DELTA       = 'delta'       in _plots
RUN_COMPARE     = 'compare'     in _plots
RUN_CROSS_PATCH = 'cross_patch' in _plots

models_to_run      = [args.model]     if args.model     else list(MODEL_CONFIGS.keys())
domains_to_run     = [args.bias]      if args.bias       else BIAS_TYPES
directions_to_run  = [args.direction] if args.direction  else list(CROSS_PATCH_CONFIGS.keys())

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
        f'Source: `{ZIP_PATH}`',
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
        f'| **High score** | Mean model probability on the correct token (clean run) |',
        f'| **Low score** | Mean probability after corrupting subject tokens |',
        f'| **Effect gap** | High − Low — how much corruption hurts; < 0.03 = low-signal |',
        f'| **Peak All/MLP/Attn** | Layer with highest NIE under each restore condition |',
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
            def nie(v): return (v - low) / gap if gap > 0 else 0.0
            nie_l0   = nie(s['states_nie'][0])
            nie_lmid = nie(s['states_nie'][mid])
            nie_last = nie(s['states_nie'][-1])
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
        nl  = 16
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
                if gap > 0:
                    nie_vals = [(v - low) / gap for v in s['states_nie']]
                else:
                    nie_vals = [0.0] * nl
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

def _extract_series(ckpt_stats_list):
    """
    From a model's all_ckpt_stats, extract per-domain series:
      labels, gap, nie_l0_states, nie_l0_mlp, nie_l0_attn
    NIE-L0 = (score_at_layer0 - low) / gap for each restore condition.
    """
    out = {d: {'labels': [], 'gap': [],
               'nie_l0_states': [], 'nie_l0_mlp': [], 'nie_l0_attn': []}
           for d in BIAS_TYPES}
    for e in ckpt_stats_list:
        for d in BIAS_TYPES:
            s = e['domains'].get(d)
            out[d]['labels'].append(e['label'])
            if s and s['effect_gap'] > 0:
                gap = s['effect_gap']
                low = s['mean_low']
                out[d]['gap'].append(gap)
                out[d]['nie_l0_states'].append((s['states_nie'][0] - low) / gap)
                out[d]['nie_l0_mlp'].append(  (s['mlp_nie'][0]    - low) / gap)
                out[d]['nie_l0_attn'].append(  (s['attn_nie'][0]   - low) / gap)
            else:
                for k in ('gap', 'nie_l0_states', 'nie_l0_mlp', 'nie_l0_attn'):
                    out[d][k].append(float('nan'))
    return out


def save_bias_delta(ckpt_stats_list, model_name, out_dir):
    """
    One PDF per domain: same 3-bar-per-layer layout as the standard bar charts,
    but Y-axis shows Δ abs. log prob diff (curr − prev checkpoint) for each layer.
    One subplot per consecutive checkpoint pair, labeled 'prev → curr'.
    """
    DELTA_LABELS = [
        'Δ Effect of single state',
        'Δ Effect with Attn severed',
        'Δ Effect with MLP severed',
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
                # 2nd bar (red)   = mlp-only restore = 'Effect with Attn severed'
                # 3rd bar (green) = attn-only restore = 'Effect with MLP severed'
                return (np.array(ckpt_s['states_nie']),
                        np.array(ckpt_s['mlp_nie']),
                        np.array(ckpt_s['attn_nie']))

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
         = (states_nie[0] − mean_low) / effect_gap. Scale-invariant across checkpoints.
      3. Raw abs. log prob diff at L0 (states_nie[0]) — absolute causal signal at first transformer layer.
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
            raw_l0 = s['states_nie'][0]
            frac_l0 = (raw_l0 - s['mean_low']) / gap if gap > 0 else float('nan')
            base_pts.append((e['label'], gap, frac_l0, raw_l0, gap < LOW_SIGNAL))

        for e in instruct_stats:
            s = e['domains'].get(domain)
            if not s:
                continue
            gap    = s['effect_gap']
            raw_l0 = s['states_nie'][0]
            frac_l0 = (raw_l0 - s['mean_low']) / gap if gap > 0 else float('nan')
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
          2nd bar (red)   = mlp-only restore = 'Effect with Attn severed'
          3rd bar (green) = attn-only restore = 'Effect with MLP severed'
        """
        return (np.array(s['states_nie']),
                np.array(s['mlp_nie']),
                np.array(s['attn_nie']))

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
            'Abs. log prob diff per layer  (blue = States, red = Attn severed, green = MLP severed)  '
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
# Data lives at CROSS_PATCH_BASE/{direction_dir}/{domain}/causal_trace/cases/
# (local filesystem, not in the zip — always read with load_npz_local).

def load_cross_patch_domain(direction_key, domain, num_sample=None):
    """
    Load cross-patch .npz files for one (direction, domain) pair.

    direction_key : key in CROSS_PATCH_CONFIGS ('pre_to_post' or 'post_to_pre')
    domain        : one of BIAS_TYPES
    num_sample    : max files per kind (None = all)

    Returns a result dict or None if no data is available:
      bias_mean, pre_blank_mean, blank_mean  — (n_layers,) subject / word arrays
      attn_mean, mlp_mean                    — Attn-only / MLP-only restore arrays
      n_cases, mean_high, mean_low           — scalar summary stats
      effect_gap                             — mean_high − mean_low
      low_sig                                — True if effect_gap < LOW_SIGNAL
      num_layer                              — number of transformer layers
    """
    cfg       = CROSS_PATCH_CONFIGS[direction_key]
    cases_dir = os.path.join(CROSS_PATCH_BASE, cfg['dir'], domain, 'causal_trace', 'cases')

    if not os.path.isdir(cases_dir):
        print(f'    [cross_patch] No directory: {cases_dir}')
        return None

    all_files = sorted(os.listdir(cases_dir))
    single_b, attn_b, mlp_b = partition_names(all_files)

    single_items = [os.path.join(cases_dir, b) for b in single_b[:num_sample]]
    attn_items   = [os.path.join(cases_dir, b) for b in attn_b[:num_sample]]
    mlp_items    = [os.path.join(cases_dir, b) for b in mlp_b[:num_sample]]

    print(f'    single={len(single_items)}, attn={len(attn_items)}, mlp={len(mlp_items)}')

    if not single_items:
        print('    No single-state files; skipping.')
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


def save_cross_patch_direction(direction_key, domain_results, out_dir):
    """
    Per-direction individual PDFs + composite figures.

    domain_results : {domain: result_dict from load_cross_patch_domain}
    out_dir        : plots/cross_patch/{direction_key}/

    Mirrors the per-checkpoint layout used for within-model tracing:
      {domain}-states.pdf      full / Attn-severed / MLP-severed bars per layer
      {domain}-words.pdf       bias-word / pre-blank / blank-token bars per layer
      composite-states.pdf     all 4 domains side-by-side
      composite-words.pdf
      composite-all.pdf        2×N grid: top=states, bottom=words
    """
    cfg   = CROSS_PATCH_CONFIGS[direction_key]
    label = cfg['label']
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


def save_cross_patch_comparison(all_direction_results, domains, out_dir):
    """
    Side-by-side comparison of both cross-patch directions for each domain.
    Y-axis is fixed across directions so the plots are directly comparable.

    all_direction_results : {direction_key: {domain: result_dict}}
    domains               : list of domains to include
    out_dir               : plots/cross_patch/

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
            fig.suptitle(
                f'Cross-patch comparison — {domain.capitalize()} bias ({plot_type})\n'
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
    name_map     = {os.path.basename(n): n for n in all_names}
    single_items = [name_map[b] for b in single_b[:num_sample] if b in name_map]
    attn_items   = [name_map[b] for b in attn_b[:num_sample]   if b in name_map]
    mlp_items    = [name_map[b] for b in mlp_b[:num_sample]    if b in name_map]
    loader       = lambda p: load_npz_zip(zf, p)

    print(f'    single={len(single_items)}, attn={len(attn_items)}, mlp={len(mlp_items)}')
    if not single_items:
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


def save_cross_patch_4panel(within_model_panels, all_direction_results, domains, out_dir):
    """
    4-panel comparison per domain (and composite over all domains):

      Panel 1 — OLMo Stage 2 last checkpoint  (within-model causal tracing)
      Panel 2 — OLMo Instruct last checkpoint  (within-model causal tracing)
      Panel 3 — Pre → Post cross-patch
      Panel 4 — Post → Pre cross-patch

    Y-axis is fixed across all 4 panels so they are directly comparable.

    within_model_panels  : {domain: {'s2_last': result_dict, 'inst_last': result_dict}}
    all_direction_results: {direction_key: {domain: result_dict}}
    domains              : ordered list of domains to include
    out_dir              : plots/cross_patch/

    Output files:
      4panel-{domain}-states.pdf
      4panel-{domain}-words.pdf
      4panel-composite-states.pdf
      4panel-composite-words.pdf
    """
    PANEL_DEFS = [
        ('s2_last',    'OLMo Stage 2\n(s2-51B)',     'within'),
        ('inst_last',  'OLMo Instruct\n(step2600)',   'within'),
        ('pre_to_post', 'Pre → Post',                 'cross'),
        ('post_to_pre', 'Post → Pre',                 'cross'),
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

    def _fill_ax(ax, res, plot_type, labels_list, title, y_min, y_max):
        if res is None:
            ax.set_visible(False)
            return
        if plot_type == 'states':
            r1, r2, r3 = res['bias_mean'], res['mlp_mean'], res['attn_mean']
        else:
            r1, r2, r3 = res['bias_mean'], res['pre_blank_mean'], res['blank_mean']
        _draw_bars(ax, r1, r2, r3, labels_list, BAR_COLORS, res['num_layer'],
                   'Layer', Y_LABEL_BARS, title)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
        if res['low_sig']:
            ax.set_facecolor(LOW_SIG_BG)
            ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                    fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

    for plot_type in ('states', 'words'):
        labels_list = STATES_LABELS if plot_type == 'states' else WORDS_LABELS

        # ── per-domain figures ────────────────────────────────────────────────
        for domain in domains:
            panels_data = [_get_res(key, src, domain) for key, _, src in PANEL_DEFS]
            y_min, y_max = _shared_ylim(panels_data, plot_type)
            if y_min is None:
                continue

            fig, axes = plt.subplots(1, 4, figsize=(FIG_BAR_W_PER_COL * 4, FIG_ROW_H))
            fig.suptitle(
                f'{domain.capitalize()} bias — 4-panel cross-patch comparison ({plot_type})\n'
                'Cols 1–2: within-model causal tracing  |  Cols 3–4: cross-model patching  '
                '|  Y-axis fixed across panels',
                fontsize=FS_SUPTITLE, fontweight='bold')
            for ax, (key, label, src), res in zip(axes, PANEL_DEFS, panels_data):
                _fill_ax(ax, res, plot_type, labels_list, label, y_min, y_max)
            plt.tight_layout()
            _savepdf(fig, os.path.join(out_dir, f'4panel-{domain}-{plot_type}.pdf'))

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
        _savepdf(fig, os.path.join(out_dir, f'4panel-composite-{plot_type}.pdf'))


# ── main loop ─────────────────────────────────────────────────────────────────

print(f'Opening zip: {ZIP_PATH}')
zf = zipfile.ZipFile(ZIP_PATH, 'r')
zip_names_all = zf.namelist()

all_models_stats = {}  # accumulate for cross-model plots

for model_name in models_to_run:
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
            use_local  = (args.source == 'local') or \
                         (args.source == 'auto' and os.path.isdir(local_dir))

            if use_local:
                all_local    = sorted(os.listdir(local_dir))
                single_b, attn_b, mlp_b = partition_names(all_local)
                single_items = [os.path.join(local_dir, b) for b in single_b[:args.num_sample]]
                attn_items   = [os.path.join(local_dir, b) for b in attn_b[:args.num_sample]]
                mlp_items    = [os.path.join(local_dir, b) for b in mlp_b[:args.num_sample]]
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
                name_map     = {os.path.basename(n): n for n in all_names}
                single_items = [name_map[b] for b in single_b[:args.num_sample] if b in name_map]
                attn_items   = [name_map[b] for b in attn_b[:args.num_sample]   if b in name_map]
                mlp_items    = [name_map[b] for b in mlp_b[:args.num_sample]    if b in name_map]
                loader       = lambda p: load_npz_zip(zf, p)
                print(f'    [zip]')

            print(f'    single={len(single_items)}, attn={len(attn_items)}, mlp={len(mlp_items)}')
            if not single_items:
                print('    No single-state files; skipping.')
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
                'states_nie':        bias_mean.tolist(),
                'attn_nie':          attn_mean.tolist(),
                'mlp_nie':           mlp_mean.tolist(),
                'pre_blank_nie':     pre_blank_mean.tolist(),
                'blank_nie':         blank_mean.tolist(),
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

    all_direction_results = {}  # {direction_key: {domain: result_dict}}

    for direction_key in directions_to_run:
        cfg_cp = CROSS_PATCH_CONFIGS[direction_key]
        print(f'\n  Direction: {cfg_cp["label"]}  ({cfg_cp["desc"]})')

        dir_out = os.path.join(cross_patch_out, direction_key)
        os.makedirs(dir_out, exist_ok=True)

        domain_results = {}
        for domain in domains_to_run:
            print(f'  {domain}')
            res = load_cross_patch_domain(direction_key, domain, args.num_sample)
            if res is not None:
                domain_results[domain] = res

        if domain_results:
            save_cross_patch_direction(direction_key, domain_results, dir_out)
            all_direction_results[direction_key] = domain_results

    # comparison plot — requires both directions to have results
    if len(all_direction_results) >= 2:
        print('\n  Generating direction-comparison plots...')
        save_cross_patch_comparison(all_direction_results, domains_to_run, cross_patch_out)
    elif len(all_direction_results) == 1:
        print('\n  Only one direction has data — skipping comparison plots.')

    # 4-panel comparison: within-model last checkpoints + both cross-patch directions
    if all_direction_results:
        print('\n  Loading within-model last checkpoints for 4-panel plots...')
        S2_CKPT   = 'stage2-ingredient3-step23852-tokens51B'
        INST_CKPT = 'step_2600'
        within_model_panels = {}
        for domain in domains_to_run:
            print(f'  {domain}')
            s2_res   = load_within_model_from_zip(
                zf, zip_names_all, BASE, 'allenai', S2_CKPT, domain, args.num_sample)
            inst_res = load_within_model_from_zip(
                zf, zip_names_all, INSTRUCT, 'allenai', INST_CKPT, domain, args.num_sample)
            within_model_panels[domain] = {'s2_last': s2_res, 'inst_last': inst_res}
        print('\n  Generating 4-panel plots...')
        save_cross_patch_4panel(within_model_panels, all_direction_results,
                                domains_to_run, cross_patch_out)

zf.close()
print('\nDone.')
