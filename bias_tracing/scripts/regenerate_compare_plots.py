"""
Regenerate comparison plots from existing stats.json files.
Does not need the zip or GPU.

Outputs (written to plots/):
  {model}/{domain}-line-checkpoints.pdf  — layer profile as lines, one per checkpoint
                                           3 panels: States / Attn-only / MLP-only
  {domain}-base-vs-instruct-lines.pdf   — same layout, base (solid) vs instruct (dashed)
                                           on the same axes for direct comparison

Usage:
    cd bias_tracing
    python scripts/regenerate_compare_plots.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from plot_utils import (
    PLOTS_BASE, BASE_MODEL, INST_MODEL, PYTHIA,
    BIAS_TYPES, LOW_SIGNAL, BAR_COLORS, Y_LABEL_BARS, Y_LABEL_NIE, STATES_LABELS,
    FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    FIG_GRID_W_PER_COL, FIG_ROW_H, FIG_LINE_W_PER_PAN, FIG_TRAJ_H,
    BASE_COLOR, INSTRUCT_COLOR, LOW_SIG_COLOR, LOW_SIG_BG,
    _draw_bars, _savepdf,
)

PANELS = [
    ('states_nie', 'Effect of single state'),
    ('mlp_nie',    'Effect with Attn severed'),
    ('attn_nie',   'Effect with MLP severed'),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _to_nie(s, key):
    """Return NIE-normalized layer array for stats entry s and data key.
    NIE = (val - mean_low) / effect_gap. Returns None when effect_gap <= 0."""
    gap = s['effect_gap']
    if gap <= 0:
        return None
    low = s['mean_low']
    return np.array([(v - low) / gap for v in s[key]])


def _load(model):
    p = os.path.join(PLOTS_BASE, model, 'stats.json')
    return json.load(open(p))['checkpoints']


def _checkpoint_colors(n, cmap_name='viridis'):
    cmap = matplotlib.colormaps[cmap_name].resampled(n)
    return [cmap(i) for i in range(n)]


# ── per-model line plot ────────────────────────────────────────────────────────

def save_line_checkpoints(ckpt_stats, model_name, out_dir, cmap='viridis'):
    """
    One PDF per domain. 3 panels (States / Attn-only / MLP-only).
    X = layer, Y = abs. log prob diff (stereo − anti).
    One colored line per checkpoint — earlier = lighter, later = darker.
    """
    n_ckpts = len(ckpt_stats)
    colors  = _checkpoint_colors(n_ckpts, cmap)

    for domain in BIAS_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(FIG_LINE_W_PER_PAN * 3, FIG_ROW_H),
                                 constrained_layout=True)
        fig.suptitle(
            f'{model_name} — {domain.capitalize()} bias: layer profile across checkpoints\n'
            'Each line = one checkpoint (light→dark = early→late training). '
            'X = layer, Y = abs. log prob diff (stereo − anti).',
            fontsize=FS_SUPTITLE, fontweight='bold')

        # collect y-range across all panels for shared axis
        all_vals = []
        for key, _ in PANELS:
            for e in ckpt_stats:
                s = e['domains'].get(domain)
                if s:
                    all_vals.extend(s[key])
        if not all_vals:
            plt.close(fig)
            continue
        y_min = min(all_vals)
        y_max = max(all_vals)
        margin = (y_max - y_min) * 0.1 or 0.01
        y_min -= margin
        y_max += margin

        for ax, (key, panel_title) in zip(axes, PANELS):
            for i, e in enumerate(ckpt_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                scores = np.array(s[key])
                low_sig = s['effect_gap'] < LOW_SIGNAL
                ls = '--' if low_sig else '-'
                lw = 1.5
                ax.plot(scores, color=colors[i], linewidth=lw, linestyle=ls,
                        label=e['label'], zorder=i + 1)
                ax.plot(0, scores[0], 'o', color=colors[i], markersize=5, zorder=i + 1)

            ax.set_title(panel_title, fontsize=FS_TITLE)
            ax.set_xlabel('Layer', fontsize=FS_LABEL)
            ax.set_ylabel(Y_LABEL_BARS, fontsize=FS_LABEL)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
            ax.grid(alpha=0.2)

        # single shared legend on the last panel
        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(handles, labels, fontsize=FS_LEGEND, loc='upper right')

        # dashed = low-signal note
        axes[0].text(0.02, 0.03, 'dashed = low-signal gap (<0.03)',
                     transform=axes[0].transAxes, fontsize=FS_ANNOT, color='gray')

        path = os.path.join(out_dir, f'{domain}-line-checkpoints.pdf')
        _savepdf(fig, path)


# ── base vs instruct side-by-side line plot ────────────────────────────────────

def save_base_vs_instruct_lines(base_stats, instruct_stats, out_dir):
    """
    One PDF per domain. 3 panels (States / Attn-only / MLP-only).
    Base checkpoints = solid lines (blue palette).
    Instruct checkpoints = dashed lines (orange palette).
    Same Y-axis across both models — direct comparison of level and shape.
    """
    n_base    = len(base_stats)
    n_instruct = len(instruct_stats)
    base_colors    = _checkpoint_colors(n_base,    'Blues')
    instruct_colors = _checkpoint_colors(n_instruct, 'Oranges')
    # make early checkpoints visible (Blues starts very light)
    base_colors    = [cm.Blues(0.35 + 0.55 * i / max(n_base - 1, 1))    for i in range(n_base)]
    instruct_colors = [cm.Oranges(0.45 + 0.45 * i / max(n_instruct - 1, 1)) for i in range(n_instruct)]

    for domain in BIAS_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(FIG_LINE_W_PER_PAN * 3, FIG_ROW_H),
                                 constrained_layout=True)
        fig.suptitle(
            f'OLMo-2-0425-1B — {domain.capitalize()} bias: Base (solid) vs Instruct (dashed)\n'
            'Each line = one checkpoint. X = layer, Y = abs. log prob diff (stereo − anti). Y-axis shared.',
            fontsize=FS_SUPTITLE, fontweight='bold')

        all_vals = []
        for key, _ in PANELS:
            for e in base_stats + instruct_stats:
                s = e['domains'].get(domain)
                if s:
                    all_vals.extend(s[key])
        if not all_vals:
            plt.close(fig)
            continue
        y_min = min(all_vals)
        y_max = max(all_vals)
        margin = (y_max - y_min) * 0.1 or 0.01
        y_min -= margin
        y_max += margin

        for ax, (key, panel_title) in zip(axes, PANELS):
            # base lines — solid
            for i, e in enumerate(base_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                scores = np.array(s[key])
                ax.plot(scores, color=base_colors[i], linewidth=1.8, linestyle='-',
                        label=f'[B] {e["label"]}', zorder=i + 1)
                ax.plot(0, scores[0], 'o', color=base_colors[i], markersize=5)

            # instruct lines — dashed
            for i, e in enumerate(instruct_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                scores = np.array(s[key])
                ax.plot(scores, color=instruct_colors[i], linewidth=2.0, linestyle='--',
                        label=f'[I] {e["label"]}', zorder=n_base + i + 1)
                ax.plot(0, scores[0], 's', color=instruct_colors[i], markersize=5)

            ax.set_title(panel_title, fontsize=FS_TITLE)
            ax.set_xlabel('Layer', fontsize=FS_LABEL)
            ax.set_ylabel(Y_LABEL_BARS, fontsize=FS_LABEL)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
            ax.grid(alpha=0.2)

        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(handles, labels, fontsize=FS_LEGEND, loc='upper right',
                        title='[B]=Base  [I]=Instruct', title_fontsize=FS_LEGEND)

        path = os.path.join(out_dir, f'{domain}-base-vs-instruct-lines.pdf')
        _savepdf(fig, path)


# ── main ──────────────────────────────────────────────────────────────────────

base_stats     = _load(BASE_MODEL)
instruct_stats = _load(INST_MODEL)
pythia_stats   = _load(PYTHIA)

COMPARE_DIR = os.path.join(PLOTS_BASE, 'compare')
os.makedirs(COMPARE_DIR, exist_ok=True)

print('=== OLMo base: line plots per domain ===')
save_line_checkpoints(base_stats, BASE_MODEL,
                      os.path.join(PLOTS_BASE, BASE_MODEL), cmap='Blues')

print('\n=== OLMo Instruct: line plots per domain ===')
save_line_checkpoints(instruct_stats, INST_MODEL,
                      os.path.join(PLOTS_BASE, INST_MODEL), cmap='Oranges')

print('\n=== Pythia: line plots per domain ===')
save_line_checkpoints(pythia_stats, PYTHIA,
                      os.path.join(PLOTS_BASE, PYTHIA), cmap='Greens')

print('\n=== Base vs Instruct: side-by-side line plots ===')
save_base_vs_instruct_lines(base_stats, instruct_stats, COMPARE_DIR)


# ── bar-chart style base vs instruct (regenerate {domain}-base-vs-instruct.pdf) ──

def save_base_vs_instruct_bars(base_stats, instruct_stats, out_dir):
    for domain in BIAS_TYPES:
        subplots = []
        last_base = base_stats[-1]
        sb = last_base['domains'].get(domain)
        if sb:
            # 2nd bar (red)   = mlp_nie = mlp-only restore = 'Effect with Attn severed'
            # 3rd bar (green) = attn_nie = attn-only restore = 'Effect with MLP severed'
            subplots.append((f'[Base] {last_base["label"]}',
                             np.array(sb['states_nie']), np.array(sb['mlp_nie']),
                             np.array(sb['attn_nie']), sb['num_layers'],
                             sb['effect_gap'] < LOW_SIGNAL))
        for e in instruct_stats:
            si = e['domains'].get(domain)
            if si:
                subplots.append((f'[Instruct] {e["label"]}',
                                 np.array(si['states_nie']), np.array(si['mlp_nie']),
                                 np.array(si['attn_nie']), si['num_layers'],
                                 si['effect_gap'] < LOW_SIGNAL))
        if not subplots:
            continue

        all_vals = np.concatenate([np.concatenate([s, a, m]) for _, s, a, m, _, _ in subplots])
        margin = (all_vals.max() - all_vals.min()) * 0.12 or 0.05
        y_min, y_max = all_vals.min() - margin, all_vals.max() + margin

        n     = len(subplots)
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(FIG_GRID_W_PER_COL * ncols, FIG_ROW_H * nrows),
                                 constrained_layout=True)
        axes_flat = axes.flatten() if n > 1 else [axes]

        fig.suptitle(
            f'OLMo-2-0425-1B — {domain.capitalize()} bias: Base vs Instruct\n'
            f'Y-axis: {Y_LABEL_BARS}. Fixed across all subplots.',
            fontsize=FS_SUPTITLE, fontweight='bold')

        for ax, (title, s, a, m, nl, low_sig) in zip(axes_flat, subplots):
            _draw_bars(ax, s, a, m, STATES_LABELS, BAR_COLORS, nl, 'Layer', Y_LABEL_BARS, title)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            if low_sig:
                ax.set_facecolor(LOW_SIG_BG)
                ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                        fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        path = os.path.join(out_dir, f'{domain}-base-vs-instruct.pdf')
        _savepdf(fig, path)


print('\n=== Base vs Instruct: bar chart plots ===')
save_base_vs_instruct_bars(base_stats, instruct_stats, COMPARE_DIR)


# ── OLMo vs Pythia cross-model line plot ──────────────────────────────────────

def save_olmo_vs_pythia(olmo_stats, pythia_stats, out_dir):
    """
    One PDF per domain. 3 panels (States / Attn-only / MLP-only).
    OLMo checkpoints = solid blue lines. Pythia checkpoints = dashed green lines.
    Same Y-axis — direct cross-architecture comparison.
    """
    n_olmo   = len(olmo_stats)
    n_pythia = len(pythia_stats)
    olmo_colors   = [cm.Blues(0.35 + 0.55 * i / max(n_olmo - 1, 1))   for i in range(n_olmo)]
    pythia_colors = [cm.Greens(0.35 + 0.55 * i / max(n_pythia - 1, 1)) for i in range(n_pythia)]

    for domain in BIAS_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(FIG_LINE_W_PER_PAN * 3, FIG_ROW_H),
                                 constrained_layout=True)
        fig.suptitle(
            f'OLMo-2-0425-1B vs Pythia-1B — {domain.capitalize()} bias: cross-architecture comparison\n'
            'Solid blue = OLMo checkpoints, dashed green = Pythia checkpoints. Y-axis shared.',
            fontsize=FS_SUPTITLE, fontweight='bold')

        all_vals = []
        for key, _ in PANELS:
            for e in olmo_stats + pythia_stats:
                s = e['domains'].get(domain)
                if s:
                    all_vals.extend(s[key])
        if not all_vals:
            plt.close(fig)
            continue
        margin = (max(all_vals) - min(all_vals)) * 0.1 or 0.01
        y_min, y_max = min(all_vals) - margin, max(all_vals) + margin

        for ax, (key, panel_title) in zip(axes, PANELS):
            for i, e in enumerate(olmo_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                scores = np.array(s[key])
                ls = '--' if s['effect_gap'] < LOW_SIGNAL else '-'
                ax.plot(scores, color=olmo_colors[i], linewidth=1.8, linestyle=ls,
                        label=f'[OLMo] {e["label"]}')
                ax.plot(0, scores[0], 'o', color=olmo_colors[i], markersize=5)

            for i, e in enumerate(pythia_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                scores = np.array(s[key])
                ls = ':' if s['effect_gap'] < LOW_SIGNAL else '--'
                ax.plot(scores, color=pythia_colors[i], linewidth=1.8, linestyle=ls,
                        label=f'[Pythia] {e["label"]}')
                ax.plot(0, scores[0], 's', color=pythia_colors[i], markersize=5)

            ax.set_title(panel_title, fontsize=FS_TITLE)
            ax.set_xlabel('Layer', fontsize=FS_LABEL)
            ax.set_ylabel(Y_LABEL_BARS, fontsize=FS_LABEL)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
            ax.grid(alpha=0.2)

        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(handles, labels, fontsize=FS_LEGEND, loc='upper right',
                        title='[OLMo]=solid  [Pythia]=dashed', title_fontsize=FS_LEGEND)
        axes[0].text(0.02, 0.03, 'dotted/dashed = low-signal (<0.03 gap)',
                     transform=axes[0].transAxes, fontsize=FS_ANNOT, color='gray')

        path = os.path.join(out_dir, f'{domain}-olmo-vs-pythia.pdf')
        _savepdf(fig, path)


print('\n=== OLMo vs Pythia: cross-architecture line plots ===')
save_olmo_vs_pythia(base_stats, pythia_stats, COMPARE_DIR)


# ── Bias trajectory (effect gap + raw L0 signal over training) ───────────────

def save_bias_trajectory(base_stats, instruct_stats, out_dir):
    """
    One PDF per domain. Three panels across the training timeline:
      Panel 1: Effect gap (high − low) — overall bias strength, raw abs log prob diff units.
      Panel 2: Embedding layer contribution — fraction of effect gap recovered at L0
               = (states_nie[0] − mean_low) / effect_gap. Scale-invariant across checkpoints.
      Panel 3: Raw abs. log prob diff at L0 (states_nie[0]) — absolute causal signal at first transformer layer.
    Base checkpoints on the left; instruct fine-tuning on the right.
    A vertical dashed line marks the phase boundary.
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

        path = os.path.join(out_dir, f'{domain}-bias-trajectory.pdf')
        _savepdf(fig, path)


print('\n=== Bias trajectory: effect gap + raw L0 signal ===')
save_bias_trajectory(base_stats, instruct_stats, COMPARE_DIR)


# ── All three models in one figure ────────────────────────────────────────────

def save_all_models_lines(base_stats, instruct_stats, pythia_stats, out_dir, use_nie=False):
    """
    One PDF per domain. 3 panels (States / Attn-only / MLP-only).
    All three model families on the same axes for direct cross-model comparison:
      OLMo base      — solid blue lines   (light → dark = early → late training)
      OLMo Instruct  — dashed orange lines
      Pythia         — dotted green lines

    Y-axis shared across panels. One line per checkpoint per model family.
    use_nie=True: Y = NIE (normalized); use_nie=False: Y = raw abs log prob diff.
    Output: {domain}-all-models[-nie].pdf
    """
    n_base     = len(base_stats)
    n_instruct = len(instruct_stats)
    n_pythia   = len(pythia_stats)

    base_colors     = [cm.Blues(0.35   + 0.55 * i / max(n_base     - 1, 1)) for i in range(n_base)]
    instruct_colors = [cm.Oranges(0.45 + 0.45 * i / max(n_instruct - 1, 1)) for i in range(n_instruct)]
    pythia_colors   = [cm.Greens(0.35  + 0.55 * i / max(n_pythia   - 1, 1)) for i in range(n_pythia)]

    LEGEND_TITLE = '[B] OLMo Base  ·  [I] OLMo Instruct  ·  [P] Pythia  |  dashed/dotted = low-signal'
    y_label      = Y_LABEL_NIE if use_nie else Y_LABEL_BARS
    fname_suffix = '-nie' if use_nie else ''

    def _get(s, key):
        return _to_nie(s, key) if use_nie else np.array(s[key])

    for domain in BIAS_TYPES:
        # extra height reserves space for the bottom legend without squishing plots
        fig, axes = plt.subplots(1, 3, figsize=(FIG_LINE_W_PER_PAN * 3, FIG_ROW_H + 1.2))
        fig.suptitle(
            f'All models — {domain.capitalize()} bias: layer profile comparison\n'
            'OLMo base (solid blue) · OLMo Instruct (dashed orange) · Pythia (dotted green). '
            f'Y = {"NIE" if use_nie else "abs log prob diff"}, shared across panels.',
            fontsize=FS_SUPTITLE, fontweight='bold')

        # collect shared y-range across all models and panels
        all_vals = []
        for key, _ in PANELS:
            for e in base_stats + instruct_stats + pythia_stats:
                s = e['domains'].get(domain)
                if s:
                    v = _get(s, key)
                    if v is not None:
                        all_vals.extend(v)
        if not all_vals:
            plt.close(fig)
            continue
        margin = (max(all_vals) - min(all_vals)) * 0.1 or 0.01
        y_min, y_max = min(all_vals) - margin, max(all_vals) + margin

        for ax, (key, panel_title) in zip(axes, PANELS):
            # OLMo base — solid
            for i, e in enumerate(base_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                vals = _get(s, key)
                if vals is None:
                    continue
                ls = '--' if s['effect_gap'] < LOW_SIGNAL else '-'
                ax.plot(vals, color=base_colors[i], linewidth=1.8,
                        linestyle=ls, label=f'[B] {e["label"]}')
                ax.plot(0, vals[0], 'o', color=base_colors[i], markersize=5)

            # OLMo Instruct — dashed
            for i, e in enumerate(instruct_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                vals = _get(s, key)
                if vals is None:
                    continue
                ls = ':' if s['effect_gap'] < LOW_SIGNAL else '--'
                ax.plot(vals, color=instruct_colors[i], linewidth=2.0,
                        linestyle=ls, label=f'[I] {e["label"]}')
                ax.plot(0, vals[0], 's', color=instruct_colors[i], markersize=5)

            # Pythia — dotted
            for i, e in enumerate(pythia_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                vals = _get(s, key)
                if vals is None:
                    continue
                ls = (0, (1, 1)) if s['effect_gap'] < LOW_SIGNAL else (0, (3, 1, 1, 1))
                ax.plot(vals, color=pythia_colors[i], linewidth=1.8,
                        linestyle=ls, label=f'[P] {e["label"]}')
                ax.plot(0, vals[0], '^', color=pythia_colors[i], markersize=5)

            ax.set_title(panel_title, fontsize=FS_TITLE)
            ax.set_xlabel('Layer', fontsize=FS_LABEL)
            ax.set_ylabel(y_label, fontsize=FS_LABEL)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
            ax.grid(alpha=0.2)

        # legend below all panels — tight_layout rect reserves the bottom strip
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   fontsize=FS_LEGEND, ncol=6, title=LEGEND_TITLE,
                   title_fontsize=FS_LEGEND, frameon=True)
        fig.tight_layout(rect=[0, 0.20, 1, 0.96])

        _savepdf(fig, os.path.join(out_dir, f'{domain}-all-models{fname_suffix}.pdf'))


print('\n=== All three models: combined line plots ===')
save_all_models_lines(base_stats, instruct_stats, pythia_stats, COMPARE_DIR)

print('\n=== All three models: combined line plots (NIE) ===')
save_all_models_lines(base_stats, instruct_stats, pythia_stats, COMPARE_DIR, use_nie=True)


# ── Sentence-pair-weighted domain average × all models ────────────────────────

def _domain_weighted_avg(ckpt_stats, key, use_nie=False):
    """
    For each checkpoint, compute a sentence-pair-weighted mean of `key`
    across all available domains:

        weighted_score[layer] = Σ(n_cases[d] * scores[d][layer]) / Σ n_cases[d]

    use_nie=True: scores are NIE-normalized before weighting; domains where
    effect_gap <= 0 are skipped entirely.
    Returns list of (label, weighted_scores, low_sig) where low_sig is True
    if any contributing domain has effect_gap < LOW_SIGNAL.
    """
    out = []
    for e in ckpt_stats:
        total_n  = 0
        weighted = None
        low_sig  = False
        for domain in BIAS_TYPES:
            s = e['domains'].get(domain)
            if not s:
                continue
            n = s['n_cases']
            if use_nie:
                scores = _to_nie(s, key)
                if scores is None:
                    continue
            else:
                scores = np.array(s[key])
            weighted = scores * n if weighted is None else weighted + scores * n
            total_n += n
            if s['effect_gap'] < LOW_SIGNAL:
                low_sig = True
        if weighted is not None and total_n > 0:
            out.append((e['label'], weighted / total_n, low_sig))
    return out


def save_domain_weighted_all_models(base_stats, instruct_stats, pythia_stats, out_dir, use_nie=False):
    """
    Single figure, 3 panels (States / Attn-only / MLP-only).
    Each line = one training checkpoint; Y = sentence-pair-weighted mean across
    all 4 bias domains (weight = n_cases per domain per checkpoint).
    use_nie=True: Y = NIE (normalized); use_nie=False: Y = raw abs log prob diff.
    OLMo base (solid blue) · OLMo Instruct (dashed orange) · Pythia (dotted green).
    Y-axis shared across panels. Legend below the figure.
    Output: domain-weighted-avg-all-models[-nie].pdf
    """
    n_base     = len(base_stats)
    n_instruct = len(instruct_stats)
    n_pythia   = len(pythia_stats)

    base_colors     = [cm.Blues(0.35   + 0.55 * i / max(n_base     - 1, 1)) for i in range(n_base)]
    instruct_colors = [cm.Oranges(0.45 + 0.45 * i / max(n_instruct - 1, 1)) for i in range(n_instruct)]
    pythia_colors   = [cm.Greens(0.35  + 0.55 * i / max(n_pythia   - 1, 1)) for i in range(n_pythia)]

    LEGEND_TITLE = ('[B] OLMo Base  ·  [I] OLMo Instruct  ·  [P] Pythia  |  '
                    'dashed/dotted = low-signal in ≥1 domain')
    y_label      = Y_LABEL_NIE if use_nie else Y_LABEL_BARS
    fname_suffix = '-nie' if use_nie else ''

    fig, axes = plt.subplots(1, 3, figsize=(FIG_LINE_W_PER_PAN * 3, FIG_ROW_H + 1.2))
    fig.suptitle(
        'All models — Sentence-pair-weighted domain average: layer profile comparison\n'
        f'Y = weighted mean {"NIE" if use_nie else "abs log prob diff"} (weight = n_cases per domain). '
        'OLMo base (solid) · OLMo Instruct (dashed) · Pythia (dotted).',
        fontsize=FS_SUPTITLE, fontweight='bold')

    for ax, (key, panel_title) in zip(axes, PANELS):
        all_vals = []

        for i, (lbl, scores, low_sig) in enumerate(_domain_weighted_avg(base_stats, key, use_nie)):
            ls = '--' if low_sig else '-'
            ax.plot(scores, color=base_colors[i], linewidth=1.8, linestyle=ls,
                    label=f'[B] {lbl}')
            ax.plot(0, scores[0], 'o', color=base_colors[i], markersize=5)
            all_vals.extend(scores)

        for i, (lbl, scores, low_sig) in enumerate(_domain_weighted_avg(instruct_stats, key, use_nie)):
            ls = ':' if low_sig else '--'
            ax.plot(scores, color=instruct_colors[i], linewidth=2.0, linestyle=ls,
                    label=f'[I] {lbl}')
            ax.plot(0, scores[0], 's', color=instruct_colors[i], markersize=5)
            all_vals.extend(scores)

        for i, (lbl, scores, low_sig) in enumerate(_domain_weighted_avg(pythia_stats, key, use_nie)):
            ls = (0, (1, 1)) if low_sig else (0, (3, 1, 1, 1))
            ax.plot(scores, color=pythia_colors[i], linewidth=1.8, linestyle=ls,
                    label=f'[P] {lbl}')
            ax.plot(0, scores[0], '^', color=pythia_colors[i], markersize=5)
            all_vals.extend(scores)

        margin = (max(all_vals) - min(all_vals)) * 0.1 if all_vals else 0.01
        ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
        ax.set_title(panel_title, fontsize=FS_TITLE)
        ax.set_xlabel('Layer', fontsize=FS_LABEL)
        ax.set_ylabel(y_label, fontsize=FS_LABEL)
        ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               fontsize=FS_LEGEND, ncol=6, title=LEGEND_TITLE,
               title_fontsize=FS_LEGEND, frameon=True)
    fig.tight_layout(rect=[0, 0.20, 1, 0.96])
    _savepdf(fig, os.path.join(out_dir, f'domain-weighted-avg-all-models{fname_suffix}.pdf'))


print('\n=== Sentence-pair-weighted domain average × all models ===')
save_domain_weighted_all_models(base_stats, instruct_stats, pythia_stats, COMPARE_DIR)

print('\n=== Sentence-pair-weighted domain average × all models (NIE) ===')
save_domain_weighted_all_models(base_stats, instruct_stats, pythia_stats, COMPARE_DIR, use_nie=True)


# ── All domains × all models in one figure ────────────────────────────────────

def save_all_domains_all_models(base_stats, instruct_stats, pythia_stats, out_dir, use_nie=False):
    """
    Single PDF: 4 rows (domains) × 3 columns (States / Attn-only / MLP-only).
    All three model families on shared axes in every cell.
    Y-axis is fixed globally across all panels.
    use_nie=True: Y = NIE (normalized); use_nie=False: Y = raw abs log prob diff.
    Output: all-domains-all-models[-nie].pdf
    """
    n_base     = len(base_stats)
    n_instruct = len(instruct_stats)
    n_pythia   = len(pythia_stats)

    base_colors     = [cm.Blues(0.35   + 0.55 * i / max(n_base     - 1, 1)) for i in range(n_base)]
    instruct_colors = [cm.Oranges(0.45 + 0.45 * i / max(n_instruct - 1, 1)) for i in range(n_instruct)]
    pythia_colors   = [cm.Greens(0.35  + 0.55 * i / max(n_pythia   - 1, 1)) for i in range(n_pythia)]

    y_label      = Y_LABEL_NIE if use_nie else Y_LABEL_BARS
    fname_suffix = '-nie' if use_nie else ''

    def _get(s, key):
        return _to_nie(s, key) if use_nie else np.array(s[key])

    n_domains = len(BIAS_TYPES)
    # extra height at bottom reserves space for the figure-level legend
    fig, axes = plt.subplots(n_domains, 3,
                             figsize=(FIG_LINE_W_PER_PAN * 3, FIG_ROW_H * n_domains + 1.2))
    fig.suptitle(
        'All domains × All models — layer profile comparison\n'
        'OLMo base (solid blue) · OLMo Instruct (dashed orange) · Pythia (dotted green). '
        f'Y = {"NIE" if use_nie else "abs log prob diff"}, fixed across all panels.',
        fontsize=FS_SUPTITLE, fontweight='bold')

    # single global y-limit across all domains and all restore conditions
    all_vals = []
    for key, _ in PANELS:
        for e in base_stats + instruct_stats + pythia_stats:
            for domain in BIAS_TYPES:
                s = e['domains'].get(domain)
                if s:
                    v = _get(s, key)
                    if v is not None:
                        all_vals.extend(v)
    margin = (max(all_vals) - min(all_vals)) * 0.1 if all_vals else 0.05
    global_ylim = (min(all_vals) - margin, max(all_vals) + margin) if all_vals else (0, 1)

    for row, domain in enumerate(BIAS_TYPES):
        for col, (key, panel_title) in enumerate(PANELS):
            ax = axes[row, col]

            # OLMo base — solid; label only in row 0 to avoid duplicate legend entries
            for i, e in enumerate(base_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                vals = _get(s, key)
                if vals is None:
                    continue
                ls  = '--' if s['effect_gap'] < LOW_SIGNAL else '-'
                lbl = f'[B] {e["label"]}' if row == 0 else '_nolegend_'
                ax.plot(vals, color=base_colors[i], linewidth=1.5, linestyle=ls, label=lbl)
                ax.plot(0, vals[0], 'o', color=base_colors[i], markersize=4)

            # OLMo Instruct — dashed
            for i, e in enumerate(instruct_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                vals = _get(s, key)
                if vals is None:
                    continue
                ls  = ':' if s['effect_gap'] < LOW_SIGNAL else '--'
                lbl = f'[I] {e["label"]}' if row == 0 else '_nolegend_'
                ax.plot(vals, color=instruct_colors[i], linewidth=1.8, linestyle=ls, label=lbl)
                ax.plot(0, vals[0], 's', color=instruct_colors[i], markersize=4)

            # Pythia — dash-dot
            for i, e in enumerate(pythia_stats):
                s = e['domains'].get(domain)
                if not s:
                    continue
                vals = _get(s, key)
                if vals is None:
                    continue
                ls  = (0, (1, 1)) if s['effect_gap'] < LOW_SIGNAL else (0, (3, 1, 1, 1))
                lbl = f'[P] {e["label"]}' if row == 0 else '_nolegend_'
                ax.plot(vals, color=pythia_colors[i], linewidth=1.5, linestyle=ls, label=lbl)
                ax.plot(0, vals[0], '^', color=pythia_colors[i], markersize=4)

            ax.set_ylim(*global_ylim)
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
            ax.grid(alpha=0.2)

            # column headers (top row only)
            if row == 0:
                ax.set_title(panel_title, fontsize=FS_TITLE, fontweight='bold')
            # row labels (left column only)
            if col == 0:
                ax.set_ylabel(f'{domain.capitalize()}\n{y_label}', fontsize=FS_LABEL)
            # x-axis label (bottom row only)
            if row == n_domains - 1:
                ax.set_xlabel('Layer', fontsize=FS_LABEL)

    # legend below the grid — tight_layout rect reserves the bottom strip
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               fontsize=FS_LEGEND, ncol=6,
               title='[B] OLMo Base  ·  [I] OLMo Instruct  ·  [P] Pythia  |  dashed/dotted = low-signal',
               title_fontsize=FS_LEGEND, frameon=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    _savepdf(fig, os.path.join(out_dir, f'all-domains-all-models{fname_suffix}.pdf'))


print('\n=== All domains × all models: combined grid ===')
save_all_domains_all_models(base_stats, instruct_stats, pythia_stats, COMPARE_DIR)

print('\n=== All domains × all models: combined grid (NIE) ===')
save_all_domains_all_models(base_stats, instruct_stats, pythia_stats, COMPARE_DIR, use_nie=True)

print('\nDone.')
