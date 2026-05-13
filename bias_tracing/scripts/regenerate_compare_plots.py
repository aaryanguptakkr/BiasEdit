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

import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PLOTS_BASE = '/deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing/plots'
BASE_MODEL  = 'OLMo-2-0425-1B'
INST_MODEL  = 'OLMo-2-0425-1B-Instruct'
PYTHIA      = 'pythia-1b'
BIAS_TYPES  = ['gender', 'profession', 'race', 'religion']
LOW_SIGNAL  = 0.03

PANELS = [
    ('states_nie', 'States (full restore)'),
    ('attn_nie',   'Attn-only'),
    ('mlp_nie',    'MLP-only'),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _load(model):
    p = os.path.join(PLOTS_BASE, model, 'stats.json')
    return json.load(open(p))['checkpoints']


def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def _checkpoint_colors(n, cmap_name='viridis'):
    cmap = cm.get_cmap(cmap_name, n)
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        fig.suptitle(
            f'{model_name} — {domain.capitalize()} bias: layer profile across checkpoints\n'
            'Each line = one checkpoint (light→dark = early→late training). '
            'X = layer, Y = abs. log prob diff (stereo − anti).',
            fontsize=11, fontweight='bold')

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

            ax.set_title(panel_title, fontsize=10)
            ax.set_xlabel('Layer', fontsize=9)
            ax.set_ylabel('Abs. log prob diff (stereo − anti)', fontsize=9)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
            ax.grid(alpha=0.2)

        # single shared legend on the last panel
        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(handles, labels, fontsize=7, loc='upper right')

        # dashed = low-signal note
        axes[0].text(0.02, 0.03, 'dashed = low-signal gap (<0.03)',
                     transform=axes[0].transAxes, fontsize=7, color='gray')

        path = os.path.join(out_dir, f'{domain}-line-checkpoints.pdf')
        _savefig(fig, path)


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
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        fig.suptitle(
            f'OLMo-2-0425-1B — {domain.capitalize()} bias: Base (solid) vs Instruct (dashed)\n'
            'Each line = one checkpoint. X = layer, Y = abs. log prob diff (stereo − anti). Y-axis shared.',
            fontsize=11, fontweight='bold')

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

            ax.set_title(panel_title, fontsize=10)
            ax.set_xlabel('Layer', fontsize=9)
            ax.set_ylabel('Abs. log prob diff (stereo − anti)', fontsize=9)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
            ax.grid(alpha=0.2)

        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(handles, labels, fontsize=7, loc='upper right',
                        title='[B]=Base  [I]=Instruct', title_fontsize=7)

        path = os.path.join(out_dir, f'{domain}-base-vs-instruct-lines.pdf')
        _savefig(fig, path)


# ── main ──────────────────────────────────────────────────────────────────────

base_stats     = _load(BASE_MODEL)
instruct_stats = _load(INST_MODEL)
pythia_stats   = _load(PYTHIA)

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
save_base_vs_instruct_lines(base_stats, instruct_stats, PLOTS_BASE)


# ── bar-chart style base vs instruct (regenerate {domain}-base-vs-instruct.pdf) ──

def _draw_bars(ax, r1, r2, r3, labels, colors, nl, xlabel, ylabel, title):
    bw = 0.25
    xs = np.arange(len(r1))
    ax.bar(xs,          r1, width=bw, color=colors[0], edgecolor='gray', label=labels[0])
    ax.bar(xs + bw,     r2, width=bw, color=colors[1], edgecolor='gray', label=labels[1])
    ax.bar(xs + 2 * bw, r3, width=bw, color=colors[2], edgecolor='gray', label=labels[2])
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6)
    ax.set_xticks(np.arange(0, nl, max(1, nl // 8)))

BAR_LABELS  = ['Effect of single state', 'Effect with Attn severed', 'Effect with MLP severed']
BAR_COLORS  = ['#2196F3', '#F44336', '#4CAF50']
Y_LABEL     = 'Abs. log prob diff (stereo − anti)'

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
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), constrained_layout=True)
        axes_flat = axes.flatten() if n > 1 else [axes]

        fig.suptitle(
            f'OLMo-2-0425-1B — {domain.capitalize()} bias: Base vs Instruct\n'
            f'Y-axis: {Y_LABEL}. Fixed across all subplots.',
            fontsize=11, fontweight='bold')

        for ax, (title, s, a, m, nl, low_sig) in zip(axes_flat, subplots):
            _draw_bars(ax, s, a, m, BAR_LABELS, BAR_COLORS, nl, 'Layer', Y_LABEL, title)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            if low_sig:
                ax.set_facecolor('#FFF9C4')
                ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                        fontsize=7, ha='right', va='top', color='#B71C1C')

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        path = os.path.join(out_dir, f'{domain}-base-vs-instruct.pdf')
        _savefig(fig, path)


print('\n=== Base vs Instruct: bar chart plots ===')
save_base_vs_instruct_bars(base_stats, instruct_stats, PLOTS_BASE)


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
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        fig.suptitle(
            f'OLMo-2-0425-1B vs Pythia-1B — {domain.capitalize()} bias: cross-architecture comparison\n'
            'Solid blue = OLMo checkpoints, dashed green = Pythia checkpoints. Y-axis shared.',
            fontsize=11, fontweight='bold')

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

            ax.set_title(panel_title, fontsize=10)
            ax.set_xlabel('Layer', fontsize=9)
            ax.set_ylabel('Abs. log prob diff (stereo − anti)', fontsize=9)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
            ax.grid(alpha=0.2)

        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(handles, labels, fontsize=6, loc='upper right',
                        title='[OLMo]=solid  [Pythia]=dashed', title_fontsize=6)
        axes[0].text(0.02, 0.03, 'dotted/dashed = low-signal (<0.03 gap)',
                     transform=axes[0].transAxes, fontsize=7, color='gray')

        path = os.path.join(out_dir, f'{domain}-olmo-vs-pythia.pdf')
        _savefig(fig, path)


print('\n=== OLMo vs Pythia: cross-architecture line plots ===')
save_olmo_vs_pythia(base_stats, pythia_stats, PLOTS_BASE)


# ── Bias trajectory (effect gap + raw L0 signal over training) ───────────────

def save_bias_trajectory(base_stats, instruct_stats, out_dir):
    """
    One PDF per domain. Three panels across the training timeline:
      Panel 1: Effect gap (high − low) — overall bias strength, raw abs log prob diff units.
      Panel 2: Embedding layer contribution — fraction of effect gap recovered at L0
               = (states_nie[0] − mean_low) / effect_gap. Scale-invariant across checkpoints.
      Panel 3: Raw abs. log prob diff at L0 (states_nie[0]) — absolute causal signal at embedding.
    Base checkpoints on the left; instruct fine-tuning on the right.
    A vertical dashed line marks the phase boundary.
    """
    BASE_COLOR     = '#1565C0'
    INSTRUCT_COLOR = '#E65100'

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
            3, 1, figsize=(max(10, n_total * 1.4), 10),
            constrained_layout=True)

        fig.suptitle(
            f'OLMo-2-0425-1B — {domain.capitalize()} bias: learning trajectory\n'
            'Left: base pre-training checkpoints   |   Right: instruction fine-tuning',
            fontsize=11, fontweight='bold')

        panel_specs = [
            (ax_gap, gaps,
             'Effect gap (high − low)',
             'Effect gap — how much corrupting subject tokens reduces bias-consistent probability\n'
             'Larger = model relies more on subject identity for this domain'),
            (ax_frac, frac_l0s,
             'NIE at L0',
             'NIE at L0 — normalized indirect effect at the embedding layer\n'
             '= (Patched − Corrupted) / (Clean − Corrupted)'),
            (ax_raw, raw_l0s,
             'Abs. log prob diff at L0\n(stereo − anti)',
             'Raw causal signal at layer 0 — absolute scale\n'
             'Reflects both embedding importance and overall bias strength'),
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
                                ha='center', fontsize=9, color='#B71C1C')

            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.25)
            ax.axhline(0, color='black', linewidth=0.7, alpha=0.3)

        path = os.path.join(out_dir, f'{domain}-bias-trajectory.pdf')
        _savefig(fig, path)


print('\n=== Bias trajectory: effect gap + raw L0 signal ===')
save_bias_trajectory(base_stats, instruct_stats, PLOTS_BASE)

print('\nDone.')
