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

import zipfile

from plot_utils import (
    PLOTS_BASE, BASE_MODEL, INST_MODEL, PYTHIA, MODEL_CONFIGS,
    BIAS_TYPES, LOW_SIGNAL, BAR_COLORS, Y_LABEL_BARS, Y_LABEL_NIE, STATES_LABELS,
    FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    FIG_GRID_W_PER_COL, FIG_ROW_H, FIG_LINE_W_PER_PAN, FIG_TRAJ_H,
    BASE_COLOR, INSTRUCT_COLOR, LOW_SIG_COLOR, LOW_SIG_BG,
    _draw_bars, _savepdf,
    collect_scores, load_npz_local, load_npz_zip, partition_names, subsample_aligned,
    zip_cases_prefix, local_cases_dir, ZIP_PATH, MAIN_ZIP,
    SCORE_METRIC, validate_score_files, normalize_stats_score_fields,
)

PANELS = [
    ('states_score', 'Effect of single state'),
    ('mlp_score',    'Effect with Attn severed'),
    ('attn_score',   'Effect with MLP severed'),
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
    data = json.load(open(p))['checkpoints']
    cfg = MODEL_CONFIGS.get(model, {})
    org = cfg.get('org', '')
    label_to_dir = {lbl: d for d, lbl in cfg.get('checkpoints', [])}
    for entry in data:
        for stats in entry['domains'].values():
            normalize_stats_score_fields(stats)
        ck_dir = label_to_dir.get(entry['label'])
        if ck_dir is None:
            continue
        for domain in BIAS_TYPES:
            if domain not in entry['domains']:
                local_d = local_cases_dir(model, org, ck_dir, domain)
                source = 'local' if os.path.isdir(local_d) else 'zip'
                stats = _npz_domain_stats(model, org, ck_dir, domain, source)
                if stats:
                    entry['domains'][domain] = stats
    return data


def _get_instruct(instruct_stats, label='step2000'):
    """Return the checkpoint entry matching label (HuggingFace main branch)."""
    for e in instruct_stats:
        if e['label'] == label:
            return e
    return instruct_stats[-1]  # fallback


def _get_pythia(pythia_stats, label='step143k'):
    """Return the Pythia end-of-training checkpoint entry."""
    for e in pythia_stats:
        if e['label'] == label:
            return e
    return pythia_stats[-1]  # fallback


# ── NPZ-based domain loader (bypasses stats.json) ─────────────────────────────
# Used by heatmap functions that need profession/race data not in stats.json.

_zip_handle = None

def _zip():
    global _zip_handle
    if _zip_handle is None:
        _zip_handle = zipfile.ZipFile(ZIP_PATH)
    return _zip_handle


_main_zip_handle = None

def _main_zip():
    global _main_zip_handle
    if _main_zip_handle is None:
        _main_zip_handle = zipfile.ZipFile(MAIN_ZIP)
    return _main_zip_handle


def _npz_domain_stats(model, org, checkpoint_dir, domain, source='zip'):
    """
    Load domain stats directly from NPZ files (all restore conditions).
    Returns a dict with the same keys as a stats.json domain entry,
    or None if no files are found.
    source='zip'      — read from ZIP_PATH
    source='main_zip' — read the OLMo 'main' checkpoint from MAIN_ZIP
                        (prefix main/{domain}/causal_trace/cases/)
    source='local'    — read from LOCAL_BASE filesystem
    """
    if source == 'main_zip':
        prefix = f'main/{domain}/causal_trace/cases/'
        zf     = _main_zip()
        names  = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith('.npz')]
        all_basenames = [os.path.basename(n) for n in names]
        single, attn, mlp = partition_names(all_basenames)
        single, attn, mlp = subsample_aligned(single, attn, mlp, None)
        single_set, attn_set, mlp_set = set(single), set(attn), set(mlp)
        file_list        = [prefix + b for b in all_basenames if b in single_set]
        file_list_attn   = [prefix + b for b in all_basenames if b in attn_set]
        file_list_mlp    = [prefix + b for b in all_basenames if b in mlp_set]
        loader = lambda p: load_npz_zip(zf, p)
    elif source == 'zip':
        prefix = zip_cases_prefix(org, model, checkpoint_dir, domain)
        zf     = _zip()
        names  = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith('.npz')]
        all_basenames = [os.path.basename(n) for n in names]
        single, attn, mlp = partition_names(all_basenames)
        single, attn, mlp = subsample_aligned(single, attn, mlp, None)
        single_set, attn_set, mlp_set = set(single), set(attn), set(mlp)
        file_list        = [prefix + b for b in all_basenames if b in single_set]
        file_list_attn   = [prefix + b for b in all_basenames if b in attn_set]
        file_list_mlp    = [prefix + b for b in all_basenames if b in mlp_set]
        loader = lambda p: load_npz_zip(zf, p)
    else:
        d = local_cases_dir(model, org, checkpoint_dir, domain)
        if not os.path.isdir(d):
            return None
        fnames = [f for f in os.listdir(d) if f.endswith('.npz')]
        single, attn, mlp = partition_names(fnames)
        single, attn, mlp = subsample_aligned(single, attn, mlp, None)
        file_list       = [os.path.join(d, f) for f in single]
        file_list_attn  = [os.path.join(d, f) for f in attn]
        file_list_mlp   = [os.path.join(d, f) for f in mlp]
        loader = load_npz_local

    if not file_list:
        return None
    expected_model = f'{org}/{model}'
    expected = {
        'score_metric': SCORE_METRIC,
        'source_model': expected_model,
        'target_model': expected_model,
        'direction': f'{expected_model} -> {expected_model}',
    }
    try:
        validate_score_files(
            (file_list, file_list_attn, file_list_mlp), loader, expected
        )
    except ValueError as ex:
        print(f'  Invalid NPZ set for {model}/{checkpoint_dir}/{domain}: {ex}')
        return None

    bm, bp, bbl, n, mh, ml = collect_scores(file_list, loader)
    if bm is None:
        return None

    # MLP-only restore (attention severed).
    mlp_score = bm.tolist()
    if file_list_mlp:
        mlp_bm, _, _, _, _, _ = collect_scores(file_list_mlp, loader)
        if mlp_bm is not None:
            mlp_score = mlp_bm.tolist()

    # Attention-only restore (MLP severed).
    attn_score = bm.tolist()
    if file_list_attn:
        attn_bm, _, _, _, _, _ = collect_scores(file_list_attn, loader)
        if attn_bm is not None:
            attn_score = attn_bm.tolist()

    return {
        'states_score':    bm.tolist(),
        'pre_blank_score': bp.tolist(),
        'blank_score':     bbl.tolist(),
        'mlp_score':       mlp_score,
        'attn_score':      attn_score,
        'mean_high': mh, 'mean_low': ml,
        'effect_gap': mh - ml,
        'n_cases': n, 'num_layers': len(bm),
    }


def _npz_for_label(model_name, label, domain):
    """
    Generic NPZ loader: looks up (org, checkpoint_dir) from MODEL_CONFIGS using
    the checkpoint label, then calls _npz_domain_stats with auto-detected source.
    Returns None if label not found or no NPZ files exist.
    """
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        return None
    org = cfg['org']
    checkpoint_dir = next((d for d, lbl in cfg['checkpoints'] if lbl == label), None)
    if checkpoint_dir is None:
        return None
    local_d = local_cases_dir(model_name, org, checkpoint_dir, domain)
    source = 'local' if os.path.isdir(local_d) else 'zip'
    return _npz_domain_stats(model_name, org, checkpoint_dir, domain, source)


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
            # 2nd bar (red) is MLP-only; 3rd (green) is attention-only.
            subplots.append((f'[Base] {last_base["label"]}',
                             np.array(sb['states_score']), np.array(sb['mlp_score']),
                             np.array(sb['attn_score']), sb['num_layers'],
                             sb['effect_gap'] < LOW_SIGNAL))
        for e in instruct_stats:
            si = e['domains'].get(domain)
            if si:
                subplots.append((f'[Instruct] {e["label"]}',
                                 np.array(si['states_score']), np.array(si['mlp_score']),
                                 np.array(si['attn_score']), si['num_layers'],
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
               = (states_score[0] − mean_low) / effect_gap. Scale-invariant across checkpoints.
      Panel 3: Raw abs. log prob diff at L0 (states_score[0]) — absolute causal signal at first transformer layer.
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
            raw_l0 = s['states_score'][0]
            frac_l0 = (raw_l0 - s['mean_low']) / gap if gap > 0 else float('nan')
            base_pts.append((e['label'], gap, frac_l0, raw_l0, gap < LOW_SIGNAL))

        for e in instruct_stats:
            s = e['domains'].get(domain)
            if not s:
                continue
            gap    = s['effect_gap']
            raw_l0 = s['states_score'][0]
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


# ── End-of-pretraining vs end-of-post-training comparison ─────────────────────

def save_prepost_comparison(base_stats, instruct_stats, out_dir):
    """
    Compare main (end of pretraining) vs step2000 (HuggingFace main branch).

    3 panels per domain — Y-axis shared across all 3 panels per row:
      Panel 1 — Abs. log prob diff : 6 grouped bars/layer, paired by condition
                  [base-states | inst-states | base-attn | inst-attn | base-mlp | inst-mlp]
                  Base = full color (blue/red/green), Instruct = light color, black edge.
      Panel 2 — Abs. log prob diff : 6 lines (solid=base, dashed=instruct)
      Panel 3 — NIE               : 6 lines (solid=base, dashed=instruct)
    Same color per condition across all 3 panels.

    Per-domain PDFs (1 row × 3 cols): prepost-{domain}.pdf
    Composite PDF   (4 rows × 3 cols): prepost-composite.pdf
    """
    base_last = base_stats[-1]
    inst_last = _get_instruct(instruct_stats)

    CONDITIONS = [
        ('states_score', BAR_COLORS[0], 'States'),
        ('mlp_score',    BAR_COLORS[1], 'Attn severed'),
        ('attn_score',   BAR_COLORS[2], 'MLP severed'),
    ]

    BAR_W    = 0.12
    TINY_GAP = 0.06   # small visual break between base and instruct groups
    PANEL_W  = FIG_LINE_W_PER_PAN

    orig_hatch_lw = plt.rcParams.get('hatch.linewidth', 1.0)
    plt.rcParams['hatch.linewidth'] = 0.7

    NIE_AXIS_COLOR = '#B35900'   # muted orange — signals independent y-axis on panel 3

    def _raw(s, key):
        return np.array(s[key]) if s else None

    def _nie(s, key):
        return _to_nie(s, key) if s else None

    def _abs_ylim(sb, si):
        """ylim from abs log prob diff values (panels 1 & 2). Bottom fixed at 0."""
        vals = []
        for key, _, _ in CONDITIONS:
            for s in [sb, si]:
                if s:
                    vals.extend(s[key])
        if not vals:
            return 0.0, 1.0
        hi = max(vals)
        return 0.0, hi * 1.12

    def _nie_ylim(sb, si):
        """ylim from NIE values (panel 3)."""
        vals = []
        for key, _, _ in CONDITIONS:
            for s in [sb, si]:
                v = _to_nie(s, key) if s else None
                if v is not None:
                    vals.extend(v.tolist())
        if not vals:
            return -0.1, 1.2
        lo, hi = min(vals), max(vals)
        span = hi - lo or 0.1
        return lo - span * 0.12, hi + span * 0.12

    def _is_low_sig(sb, si):
        return (sb is not None and sb['effect_gap'] < LOW_SIGNAL) or \
               (si is not None and si['effect_gap'] < LOW_SIGNAL)

    def _annotate_low_sig(ax):
        ax.set_facecolor(LOW_SIG_BG)
        ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

    def _draw_6bars(ax, sb, si, ylabel, title):
        """[base×3 | tiny gap | instruct×3] per layer. Instruct = hatch on same color."""
        nl = (sb or si)['num_layers']
        xs = np.arange(nl)
        n  = len(CONDITIONS)
        # base group centred left of zero, instruct group to the right with tiny gap
        base_start = -(n * BAR_W + TINY_GAP / 2)
        inst_start =  TINY_GAP / 2

        for ci, (key, color, cond_label) in enumerate(CONDITIONS):
            bv = _raw(sb, key)
            iv = _raw(si, key)
            b_off = base_start + ci * BAR_W + BAR_W / 2
            i_off = inst_start + ci * BAR_W + BAR_W / 2
            if bv is not None:
                ax.bar(xs + b_off, bv, width=BAR_W, color=color,
                       edgecolor='black', linewidth=0.7,
                       label=f'Base – {cond_label}')
            if iv is not None:
                ax.bar(xs + i_off, iv, width=BAR_W, color=color,
                       edgecolor='black', linewidth=0.7,
                       hatch='////',
                       label=f'Instruct – {cond_label}')

        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlabel('Layer', fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=FS_TICK)
        ax.grid(axis='y', alpha=0.2)
        if _is_low_sig(sb, si):
            _annotate_low_sig(ax)

    def _draw_6lines(ax, sb, si, use_nie, ylabel, title, nie_axis=False):
        """6 lines: 3 conditions × 2 models. Solid = base, dashed = instruct."""
        nl = (sb or si)['num_layers']
        xs = np.arange(nl)

        for key, color, cond_label in CONDITIONS:
            bv = _nie(sb, key) if use_nie else _raw(sb, key)
            iv = _nie(si, key) if use_nie else _raw(si, key)
            if bv is not None:
                ax.plot(xs, bv, color=color, linewidth=1.8, linestyle='-',
                        marker='o', markersize=3,
                        label=f'Base – {cond_label}')
            if iv is not None:
                ax.plot(xs, iv, color=color, linewidth=1.8, linestyle='--',
                        marker='s', markersize=3,
                        label=f'Instruct – {cond_label}')

        ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlabel('Layer', fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=FS_TICK)
        ax.grid(alpha=0.2)
        if nie_axis:
            # color spine, ticks, and labels to signal independent y-axis
            for spine in ['left']:
                ax.spines[spine].set_color(NIE_AXIS_COLOR)
            ax.yaxis.label.set_color(NIE_AXIS_COLOR)
            ax.tick_params(axis='y', colors=NIE_AXIS_COLOR)
        if _is_low_sig(sb, si):
            _annotate_low_sig(ax)

    def _build_fig(domains, suptitle):
        nrows  = len(domains)
        fig, axes = plt.subplots(
            nrows, 3,
            figsize=(PANEL_W * 3, FIG_ROW_H * nrows),
            gridspec_kw={'width_ratios': [4, 3, 3]},
        )
        if nrows == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(suptitle, fontsize=FS_SUPTITLE, fontweight='bold')

        letters = [chr(ord('A') + row * 3 + col) for row in range(nrows) for col in range(3)]

        for row, domain in enumerate(domains):
            sb = _npz_for_label(BASE_MODEL, base_last['label'], domain)
            si = _npz_for_label(INST_MODEL, inst_last['label'], domain)
            if sb is None and si is None:
                for col in range(3):
                    axes[row, col].set_visible(False)
                continue
            abs_ylim = _abs_ylim(sb, si)
            nie_ylim = _nie_ylim(sb, si)

            _draw_6bars(axes[row, 0], sb, si, Y_LABEL_BARS,
                        f'{domain.capitalize()} — Abs. log prob diff (bars)')
            axes[row, 0].set_ylim(*abs_ylim)

            _draw_6lines(axes[row, 1], sb, si, False, Y_LABEL_BARS,
                         f'{domain.capitalize()} — Abs. log prob diff (lines)')
            axes[row, 1].set_ylim(*abs_ylim)

            _draw_6lines(axes[row, 2], sb, si, True, Y_LABEL_NIE,
                         f'{domain.capitalize()} — NIE (lines)', nie_axis=True)
            axes[row, 2].set_ylim(*nie_ylim)

            for col in range(3):
                axes[row, col].text(
                    0.01, 0.99, letters[row * 3 + col],
                    transform=axes[row, col].transAxes,
                    fontsize=11, fontweight='bold', va='top', ha='left',
                    zorder=10)

        h_bar, l_bar   = axes[0, 0].get_legend_handles_labels()
        h_line, l_line = axes[0, 2].get_legend_handles_labels()
        fig.legend(h_bar + h_line, l_bar + l_line,
                   loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4,
                   fontsize=FS_LEGEND, frameon=True,
                   title='Panel 1: solid=Base, hatched=Instruct  |  Panels 2–3: solid=Base, dashed=Instruct'
                         f'  |  {NIE_AXIS_COLOR} axis = independent NIE scale',
                   title_fontsize=FS_LEGEND)
        fig.tight_layout(rect=[0, 0.10, 1, 0.96])
        return fig

    SUPTITLE = (
        'End of pretraining (main) vs post-training (step2000, HuggingFace main)\n'
        'Panel 1: Abs. log prob diff (bars)  |  Panel 2: Abs. log prob diff (lines)  |  Panel 3: NIE (independent y-axis)\n'
        'Solid = Base, hatched/dashed = Instruct. Panels 1 & 2 share y-axis.'
    )

    for domain in BIAS_TYPES:
        fig = _build_fig([domain], f'{domain.capitalize()} bias — {SUPTITLE}')
        _savepdf(fig, os.path.join(out_dir, f'prepost-{domain}.pdf'))

    fig = _build_fig(BIAS_TYPES, SUPTITLE)
    _savepdf(fig, os.path.join(out_dir, 'prepost-composite.pdf'))

    plt.rcParams['hatch.linewidth'] = orig_hatch_lw


print('\n=== End-of-pretraining vs end-of-post-training comparison ===')
save_prepost_comparison(base_stats, instruct_stats, COMPARE_DIR)


# ── Cross-patch NIE comparison ─────────────────────────────────────────────────

def save_cross_patch_nie_comparison(base_stats, instruct_stats, out_dir):
    """
    4-line NIE plot per domain + composite:
      Within-model Base  (blue)   — main within-model causal tracing
      Within-model Inst  (green)  — step2600 within-model causal tracing
      Pre → Post         (red)    — base activations patched into instruct model
      Post → Pre         (orange) — instruct activations patched into base model
    All solid lines. NIE normalized to each target model's own effect gap.
    Outputs: {domain}-cross-patch-nie.pdf, cross-patch-nie-composite.pdf
    """
    from plot_utils import (CROSS_PATCH_BASE, CROSS_PATCH_CONFIGS,
                            collect_scores, load_npz_local, partition_names)

    LINE_COLORS = ['#1565C0', '#2E7D32', '#C62828', '#E65100']
    LINE_LABELS = [
        'Within-model Base (main)',
        'Within-model Instruct (step2000)',
        'Pre → Post  (base acts. → instruct)',
        'Post → Pre  (instruct acts. → base)',
    ]

    base_last = base_stats[-1]
    inst_last = _get_instruct(instruct_stats)

    def _cp_nie(direction, domain):
        cp_dir = os.path.join(CROSS_PATCH_BASE,
                              CROSS_PATCH_CONFIGS[direction]['dir'],
                              domain, 'causal_trace', 'cases')
        if not os.path.isdir(cp_dir):
            return None
        fnames = [f for f in os.listdir(cp_dir) if f.endswith('.npz')]
        single, _, _ = partition_names(fnames)
        file_list = [os.path.join(cp_dir, f) for f in single]
        if not file_list:
            return None
        bias_mean, _, _, _, mean_high, mean_low = collect_scores(file_list, load_npz_local)
        if bias_mean is None:
            return None
        gap = mean_high - mean_low
        return (bias_mean - mean_low) / gap if gap > 0 else None

    def _domain_nies(domain):
        sb = _npz_for_label(BASE_MODEL, base_last['label'], domain)
        si = _npz_for_label(INST_MODEL, inst_last['label'], domain)
        return [
            _to_nie(sb, 'states_score') if sb else None,
            _to_nie(si, 'states_score') if si else None,
            _cp_nie('pre_to_post', domain),
            _cp_nie('post_to_pre', domain),
        ]

    def _draw_domain(ax, domain, nies, show_ylabel=True):
        all_vals = []
        n_layers = 16
        for nie, color, label in zip(nies, LINE_COLORS, LINE_LABELS):
            if nie is None:
                continue
            n_layers = len(nie)
            ax.plot(np.arange(n_layers), nie,
                    color=color, linewidth=2.0, linestyle='-',
                    marker='o', markersize=4, label=label)
            all_vals.extend(nie.tolist())

        ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        ax.set_title(domain.capitalize(), fontsize=FS_TITLE)
        ax.set_xlabel('Layer', fontsize=FS_LABEL)
        if show_ylabel:
            ax.set_ylabel(Y_LABEL_NIE, fontsize=FS_LABEL)
        tick_step = max(1, n_layers // 8)
        ax.set_xticks(np.arange(0, n_layers, tick_step))
        ax.grid(alpha=0.2)
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            span = hi - lo or 0.1
            ax.set_ylim(lo - span * 0.12, hi + span * 0.12)

    # per-domain PDFs
    for domain in BIAS_TYPES:
        nies = _domain_nies(domain)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle(
            f'{domain.capitalize()} bias — NIE: within-model vs cross-patch\n'
            "NIE normalized to each target model's own effect gap.",
            fontsize=FS_SUPTITLE, fontweight='bold')
        _draw_domain(ax, domain, nies)
        ax.legend(fontsize=FS_LEGEND, loc='upper right')
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        _savepdf(fig, os.path.join(out_dir, f'{domain}-cross-patch-nie.pdf'))

    # composite: 1 row × 4 domains
    fig, axes = plt.subplots(1, len(BIAS_TYPES),
                             figsize=(FIG_LINE_W_PER_PAN * len(BIAS_TYPES), FIG_ROW_H))
    fig.suptitle(
        'NIE comparison: within-model vs cross-patch — all domains\n'
        "NIE normalized to each target model's own effect gap.",
        fontsize=FS_SUPTITLE, fontweight='bold')

    all_domain_nies = {}
    for ax, domain in zip(axes, BIAS_TYPES):
        nies = _domain_nies(domain)
        all_domain_nies[domain] = nies
        _draw_domain(ax, domain, nies, show_ylabel=(domain == BIAS_TYPES[0]))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=FS_LEGEND, frameon=True)
    fig.tight_layout(rect=[0, 0.18, 1, 0.94])
    _savepdf(fig, os.path.join(out_dir, 'cross-patch-nie-composite.pdf'))

    # summary table
    print('\n  Peak NIE and peak layer per condition:')
    short = ['WithinBase', 'WithinInst', 'Pre→Post', 'Post→Pre']
    print(f'  {"Domain":<12}' + ''.join(f'  {h:>18}' for h in short))
    print('  ' + '-' * (12 + 20 * 4))
    for domain in BIAS_TYPES:
        row = f'  {domain:<12}'
        for nie in all_domain_nies[domain]:
            if nie is None:
                row += f'  {"N/A":>18}'
            else:
                row += f'  {float(np.max(nie)):.3f} @ L{int(np.argmax(nie)):<12}'
        print(row)


print('\n=== Cross-patch NIE comparison ===')
save_cross_patch_nie_comparison(base_stats, instruct_stats, COMPARE_DIR)

# ── NIE heatmap (Figure 2 clone) ──────────────────────────────────────────────

def _mark_K_cols(ax, n_layers):
    """
    Visually mark K = {subject, target} on a 3-column heatmap axes.
    - Shades the non-K pre-target column (col 1) with a light gray overlay.
    - Draws a dashed border around the K columns (cols 0 and 2).
    """
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.5, -0.5), 1.0, n_layers,
                            facecolor='gray', alpha=0.20, zorder=4, clip_on=True,
                            linewidth=0))
    for col in [0, 2]:
        ax.add_patch(Rectangle((col - 0.5, -0.5), 1.0, n_layers,
                                linewidth=1.8, edgecolor='#222222', facecolor='none',
                                linestyle='--', zorder=5, clip_on=True))


def save_nie_heatmap(base_stats, instruct_stats, out_dir):
    """
    NIE heatmap: Base (main) | Instruct (step2000) | Difference (Inst − Base)
    X: token position category — [subject], [pre-target], [target]
    Y: layer 0–15, L0 at top (closest to input)
    Colormap: RdBu_r — red = positive NIE (causal signal), blue = negative.
    Base and Instruct panels clamped to [−1, 1].
    Difference panel uses symmetric ±max|diff| range.
    K columns (subject, target) outlined with dashed border; pre-target shaded gray.
    Outputs: {domain}-nie-heatmap.pdf, nie-heatmap-composite.pdf
    """
    POS_LABELS = ['[subject] (K)', '[pre-target]', '[target] (K)']
    N_LAYERS   = 16

    def _mat(s):
        lo, gap = s['mean_low'], s['effect_gap']
        if gap <= 0:
            return None
        return np.column_stack([
            (np.array(s['states_score'])    - lo) / gap,
            (np.array(s['pre_blank_score']) - lo) / gap,
            (np.array(s['blank_score'])     - lo) / gap,
        ])  # (16, 3)

    def _draw_row(row_axes, domain):
        ax0, ax1, ax2 = row_axes
        sb = _npz_domain_stats('OLMo-2-0425-1B', 'allenai',
                               'main', domain, 'main_zip')
        si = _npz_domain_stats('OLMo-2-0425-1B-Instruct', 'allenai',
                               'step_2000', domain, 'local')
        mb = _mat(sb) if sb else None
        mi = _mat(si) if si else None
        if mb is None or mi is None:
            for ax in [ax0, ax1, ax2]:
                ax.set_visible(False)
            return None

        md   = mi - mb
        dlim = max(0.05, float(np.max(np.abs(md))))
        kw   = dict(aspect='auto', cmap='RdBu_r', origin='upper')

        im0 = ax0.imshow(np.clip(mb, -1, 1), vmin=-1, vmax=1, **kw)
        ax1.imshow(np.clip(mi, -1, 1),        vmin=-1, vmax=1, **kw)
        im2 = ax2.imshow(md,                  vmin=-dlim, vmax=dlim, **kw)

        for ax, ttl in zip([ax0, ax1, ax2],
                           ['Base (main)', 'Instruct (step2000)', 'Inst − Base']):
            ax.set_title(f'{domain.capitalize()} — {ttl}', fontsize=FS_TITLE)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(POS_LABELS, fontsize=FS_TICK, rotation=20, ha='right')
            ax.set_yticks(range(0, N_LAYERS, 2))
            ax.set_yticklabels(range(0, N_LAYERS, 2), fontsize=FS_TICK)
            _mark_K_cols(ax, N_LAYERS)
        ax0.set_ylabel('Layer', fontsize=FS_LABEL)
        return im0, im2

    # ── per-domain PDFs ────────────────────────────────────────────────────────
    for domain in BIAS_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), constrained_layout=True)
        fig.suptitle(
            f'{domain.capitalize()} — NIE heatmap  |  Base vs Instruct\n'
            'Red = positive NIE (causal signal), blue = negative. '
            'Base/Instruct clamped [−1, 1]. Difference: red = instruct gained signal.',
            fontsize=FS_SUPTITLE, fontweight='bold')
        result = _draw_row(axes, domain)
        if result:
            im0, im2 = result
            fig.colorbar(im0, ax=[axes[0], axes[1]],
                         label='NIE (clamped [−1, 1])', shrink=0.8)
            fig.colorbar(im2, ax=axes[2],
                         label='ΔNIE (Inst − Base)', shrink=0.8)
        _savepdf(fig, os.path.join(out_dir, f'{domain}-nie-heatmap.pdf'))

    # ── composite: 3 domains × 3 panels ───────────────────────────────────────
    active = ['gender', 'profession', 'race']
    nr = len(active)
    fig, axes = plt.subplots(nr, 3, figsize=(12, 5 * nr), constrained_layout=True)
    fig.suptitle(
        'NIE heatmap — all domains  |  Base (main) vs Instruct (step2000)\n'
        'Red = positive NIE. Base/Instruct clamped [−1, 1]. Difference per-domain scaled.',
        fontsize=FS_SUPTITLE, fontweight='bold')
    for row, domain in enumerate(active):
        result = _draw_row(axes[row], domain)
        if result:
            im0, im2 = result
            fig.colorbar(im0, ax=[axes[row, 0], axes[row, 1]],
                         label='NIE', shrink=0.8)
            fig.colorbar(im2, ax=axes[row, 2],
                         label='ΔNIE', shrink=0.8)
    _savepdf(fig, os.path.join(out_dir, 'nie-heatmap-composite.pdf'))


# ── AbsLogProb heatmap (amplification visual) ─────────────────────────────────

def save_alp_heatmap(base_stats, instruct_stats, out_dir):
    """
    AbsLogProb heatmap: Base (main) | Instruct (step2000) | Difference
    Raw patched scores (abs log prob diff at each position×layer) — not NIE-normalized.
    X: [subject], [pre-target], [target]
    Y: layer 0–15, L0 at top
    Main panels: YlOrRd (shared vmin/vmax per domain).
    Difference panel: RdBu_r — red = instruct amplified, blue = decreased.
    Outputs: {domain}-alp-heatmap.pdf, alp-heatmap-composite.pdf
    """
    POS_LABELS = ['[subject] (K)', '[pre-target]', '[target] (K)']
    N_LAYERS   = 16

    def _mat(s):
        return np.column_stack([
            np.array(s['states_score']),
            np.array(s['pre_blank_score']),
            np.array(s['blank_score']),
        ])  # (16, 3) — raw patched scores

    def _draw_row(row_axes, domain):
        ax0, ax1, ax2 = row_axes
        sb = _npz_domain_stats('OLMo-2-0425-1B', 'allenai',
                               'main', domain, 'main_zip')
        si = _npz_domain_stats('OLMo-2-0425-1B-Instruct', 'allenai',
                               'step_2000', domain, 'local')
        if not sb or not si:
            for ax in [ax0, ax1, ax2]:
                ax.set_visible(False)
            return None

        mb   = _mat(sb)
        mi   = _mat(si)
        md   = mi - mb
        vmax = max(float(np.max(mb)), float(np.max(mi)))
        vmin = min(0.0, float(np.min(mb)), float(np.min(mi)))
        dlim = max(0.01, float(np.max(np.abs(md))))

        im0 = ax0.imshow(mb, aspect='auto', cmap='YlOrRd', origin='upper',
                         vmin=vmin, vmax=vmax)
        ax1.imshow(mi, aspect='auto', cmap='YlOrRd', origin='upper',
                   vmin=vmin, vmax=vmax)
        im2 = ax2.imshow(md, aspect='auto', cmap='RdBu_r', origin='upper',
                         vmin=-dlim, vmax=dlim)

        for ax, ttl in zip([ax0, ax1, ax2],
                           ['Base (main)', 'Instruct (step2000)', 'Inst − Base']):
            ax.set_title(f'{domain.capitalize()} — {ttl}', fontsize=FS_TITLE)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(POS_LABELS, fontsize=FS_TICK, rotation=20, ha='right')
            ax.set_yticks(range(0, N_LAYERS, 2))
            ax.set_yticklabels(range(0, N_LAYERS, 2), fontsize=FS_TICK)
            _mark_K_cols(ax, N_LAYERS)
        ax0.set_ylabel('Layer', fontsize=FS_LABEL)
        return im0, im2

    # ── per-domain PDFs ────────────────────────────────────────────────────────
    for domain in BIAS_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), constrained_layout=True)
        fig.suptitle(
            f'{domain.capitalize()} — AbsLogProb heatmap  |  Base vs Instruct\n'
            'Abs. log prob diff (stereo − anti) at each (position, layer). '
            'YlOrRd: yellow=low, red=high. Difference: red = instruct amplified signal.',
            fontsize=FS_SUPTITLE, fontweight='bold')
        result = _draw_row(axes, domain)
        if result:
            im0, im2 = result
            fig.colorbar(im0, ax=[axes[0], axes[1]],
                         label='Abs. log prob diff (stereo − anti)', shrink=0.8)
            fig.colorbar(im2, ax=axes[2],
                         label='Δ Abs. log prob diff (Inst − Base)', shrink=0.8)
        _savepdf(fig, os.path.join(out_dir, f'{domain}-alp-heatmap.pdf'))

    # ── composite: 3 domains × 3 panels ───────────────────────────────────────
    active = ['gender', 'profession', 'race']
    nr = len(active)
    fig, axes = plt.subplots(nr, 3, figsize=(12, 5 * nr), constrained_layout=True)
    fig.suptitle(
        'AbsLogProb heatmap — all domains  |  Base (main) vs Instruct (step2000)\n'
        'Raw patched scores. YlOrRd for main panels. Difference in RdBu_r (red = amplified).',
        fontsize=FS_SUPTITLE, fontweight='bold')
    for row, domain in enumerate(active):
        result = _draw_row(axes[row], domain)
        if result:
            im0, im2 = result
            fig.colorbar(im0, ax=[axes[row, 0], axes[row, 1]],
                         label='Abs. log prob diff', shrink=0.8)
            fig.colorbar(im2, ax=axes[row, 2],
                         label='Δ Abs. log prob diff', shrink=0.8)
    _savepdf(fig, os.path.join(out_dir, 'alp-heatmap-composite.pdf'))


print('\n=== NIE heatmap (Base vs Instruct) ===')
save_nie_heatmap(base_stats, instruct_stats, COMPARE_DIR)

print('\n=== AbsLogProb heatmap (amplification) ===')
save_alp_heatmap(base_stats, instruct_stats, COMPARE_DIR)


# ── OLMo main vs Pythia step143k comparison ─────────────────────────────────

def save_olmo_vs_pythia_prepost(base_stats, pythia_stats, out_dir):
    """
    Compare main (OLMo end of pretraining) vs Pythia step143k (end of training).
    Same 3-panel layout as save_prepost_comparison: bars + ALP lines + NIE lines.
    Per-domain PDFs: olmo-pythia-{domain}.pdf
    Composite PDF:   olmo-pythia-composite.pdf
    """
    base_last   = base_stats[-1]
    pythia_last = _get_pythia(pythia_stats)

    CONDITIONS = [
        ('states_score', BAR_COLORS[0], 'States'),
        ('mlp_score',    BAR_COLORS[1], 'Attn severed'),
        ('attn_score',   BAR_COLORS[2], 'MLP severed'),
    ]

    BAR_W    = 0.12
    TINY_GAP = 0.06
    PANEL_W  = FIG_LINE_W_PER_PAN
    NIE_AXIS_COLOR = '#005C27'   # dark green — signals independent y-axis on panel 3

    orig_hatch_lw = plt.rcParams.get('hatch.linewidth', 1.0)
    plt.rcParams['hatch.linewidth'] = 0.7

    def _raw(s, key):
        return np.array(s[key]) if s else None

    def _nie_v(s, key):
        return _to_nie(s, key) if s else None

    def _abs_ylim(sb, sp):
        vals = []
        for key, _, _ in CONDITIONS:
            for s in [sb, sp]:
                if s:
                    vals.extend(s[key])
        return (0.0, max(vals) * 1.12) if vals else (0.0, 1.0)

    def _nie_ylim(sb, sp):
        vals = []
        for key, _, _ in CONDITIONS:
            for s in [sb, sp]:
                v = _to_nie(s, key) if s else None
                if v is not None:
                    vals.extend(v.tolist())
        if not vals:
            return -0.1, 1.2
        lo, hi = min(vals), max(vals)
        span = hi - lo or 0.1
        return lo - span * 0.12, hi + span * 0.12

    def _is_low_sig(sb, sp):
        return (sb is not None and sb['effect_gap'] < LOW_SIGNAL) or \
               (sp is not None and sp['effect_gap'] < LOW_SIGNAL)

    def _annotate_low_sig(ax):
        ax.set_facecolor(LOW_SIG_BG)
        ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                fontsize=FS_ANNOT, ha='right', va='top', color=LOW_SIG_COLOR)

    def _draw_6bars(ax, sb, sp, ylabel, title):
        nl = (sb or sp)['num_layers']
        xs = np.arange(nl)
        n  = len(CONDITIONS)
        base_start = -(n * BAR_W + TINY_GAP / 2)
        pyth_start =  TINY_GAP / 2
        for ci, (key, color, cond_label) in enumerate(CONDITIONS):
            bv = _raw(sb, key)
            pv = _raw(sp, key)
            b_off = base_start + ci * BAR_W + BAR_W / 2
            p_off = pyth_start + ci * BAR_W + BAR_W / 2
            if bv is not None:
                ax.bar(xs + b_off, bv, width=BAR_W, color=color,
                       edgecolor='black', linewidth=0.7,
                       label=f'OLMo – {cond_label}')
            if pv is not None:
                ax.bar(xs + p_off, pv, width=BAR_W, color=color,
                       edgecolor='black', linewidth=0.7, hatch='////',
                       label=f'Pythia – {cond_label}')
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlabel('Layer', fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=FS_TICK)
        ax.grid(axis='y', alpha=0.2)
        if _is_low_sig(sb, sp):
            _annotate_low_sig(ax)

    def _draw_6lines(ax, sb, sp, use_nie, ylabel, title, nie_axis=False):
        nl = (sb or sp)['num_layers']
        xs = np.arange(nl)
        for key, color, cond_label in CONDITIONS:
            bv = _nie_v(sb, key) if use_nie else _raw(sb, key)
            pv = _nie_v(sp, key) if use_nie else _raw(sp, key)
            if bv is not None:
                ax.plot(xs, bv, color=color, linewidth=1.8, linestyle='-',
                        marker='o', markersize=3, label=f'OLMo – {cond_label}')
            if pv is not None:
                ax.plot(xs, pv, color=color, linewidth=1.8, linestyle='--',
                        marker='s', markersize=3, label=f'Pythia – {cond_label}')
        ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlabel('Layer', fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=FS_TICK)
        ax.grid(alpha=0.2)
        if nie_axis:
            ax.spines['left'].set_color(NIE_AXIS_COLOR)
            ax.yaxis.label.set_color(NIE_AXIS_COLOR)
            ax.tick_params(axis='y', colors=NIE_AXIS_COLOR)
        if _is_low_sig(sb, sp):
            _annotate_low_sig(ax)

    def _build_fig(domains, suptitle):
        nrows = len(domains)
        fig, axes = plt.subplots(
            nrows, 3,
            figsize=(PANEL_W * 3, FIG_ROW_H * nrows),
            gridspec_kw={'width_ratios': [4, 3, 3]},
        )
        if nrows == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(suptitle, fontsize=FS_SUPTITLE, fontweight='bold')

        for row, domain in enumerate(domains):
            sb = base_last['domains'].get(domain)
            sp = pythia_last['domains'].get(domain)
            if sb is None and sp is None:
                for col in range(3):
                    axes[row, col].set_visible(False)
                continue
            abs_ylim = _abs_ylim(sb, sp)
            nie_ylim = _nie_ylim(sb, sp)

            _draw_6bars(axes[row, 0], sb, sp, Y_LABEL_BARS,
                        f'{domain.capitalize()} — Abs. log prob diff (bars)')
            axes[row, 0].set_ylim(*abs_ylim)

            _draw_6lines(axes[row, 1], sb, sp, False, Y_LABEL_BARS,
                         f'{domain.capitalize()} — Abs. log prob diff (lines)')
            axes[row, 1].set_ylim(*abs_ylim)

            _draw_6lines(axes[row, 2], sb, sp, True, Y_LABEL_NIE,
                         f'{domain.capitalize()} — NIE (lines)', nie_axis=True)
            axes[row, 2].set_ylim(*nie_ylim)

        h_bar, l_bar   = axes[0, 0].get_legend_handles_labels()
        h_line, l_line = axes[0, 2].get_legend_handles_labels()
        fig.legend(h_bar + h_line, l_bar + l_line,
                   loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4,
                   fontsize=FS_LEGEND, frameon=True,
                   title='Panel 1: solid=OLMo, hatched=Pythia  |  Panels 2–3: solid=OLMo, dashed=Pythia'
                         f'  |  {NIE_AXIS_COLOR} axis = independent NIE scale',
                   title_fontsize=FS_LEGEND)
        fig.tight_layout(rect=[0, 0.10, 1, 0.96])
        return fig

    SUPTITLE = (
        'OLMo-2-0425-1B (main) vs Pythia-1B (step143k) — cross-architecture pre-training comparison\n'
        'Panel 1: Abs. log prob diff (bars)  |  Panel 2: Abs. log prob diff (lines)  |  Panel 3: NIE (independent y-axis)\n'
        'Solid = OLMo, hatched/dashed = Pythia. Panels 1 & 2 share y-axis.'
    )

    for domain in BIAS_TYPES:
        fig = _build_fig([domain], f'{domain.capitalize()} bias — {SUPTITLE}')
        _savepdf(fig, os.path.join(out_dir, f'olmo-pythia-{domain}.pdf'))

    fig = _build_fig(BIAS_TYPES, SUPTITLE)
    _savepdf(fig, os.path.join(out_dir, 'olmo-pythia-composite.pdf'))

    plt.rcParams['hatch.linewidth'] = orig_hatch_lw


def save_olmo_pythia_nie_heatmap(base_stats, pythia_stats, out_dir):
    """
    NIE heatmap: OLMo (main) | Pythia (step143k) | Difference (Pythia − OLMo)
    Same layout as save_nie_heatmap.
    Outputs: {domain}-olmo-pythia-nie-heatmap.pdf, olmo-pythia-nie-heatmap-composite.pdf
    """
    base_last   = base_stats[-1]
    pythia_last = _get_pythia(pythia_stats)
    POS_LABELS  = ['[subject]', '[pre-target]', '[target]']
    N_LAYERS    = 16

    def _mat(s):
        lo, gap = s['mean_low'], s['effect_gap']
        if gap <= 0:
            return None
        return np.column_stack([
            (np.array(s['states_score'])    - lo) / gap,
            (np.array(s['pre_blank_score']) - lo) / gap,
            (np.array(s['blank_score'])     - lo) / gap,
        ])

    def _draw_row(row_axes, domain):
        ax0, ax1, ax2 = row_axes
        sb = base_last['domains'].get(domain)
        sp = pythia_last['domains'].get(domain)
        mb = _mat(sb) if sb else None
        mp = _mat(sp) if sp else None
        if mb is None or mp is None:
            for ax in [ax0, ax1, ax2]:
                ax.set_visible(False)
            return None
        md   = mp - mb
        dlim = max(0.05, float(np.max(np.abs(md))))
        kw   = dict(aspect='auto', cmap='RdBu_r', origin='upper')
        im0 = ax0.imshow(np.clip(mb, -1, 1), vmin=-1, vmax=1, **kw)
        ax1.imshow(np.clip(mp, -1, 1),        vmin=-1, vmax=1, **kw)
        im2 = ax2.imshow(md,                  vmin=-dlim, vmax=dlim, **kw)
        for ax, ttl in zip([ax0, ax1, ax2],
                           ['OLMo (main)', 'Pythia (step143k)', 'Pythia − OLMo']):
            ax.set_title(f'{domain.capitalize()} — {ttl}', fontsize=FS_TITLE)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(POS_LABELS, fontsize=FS_TICK, rotation=15, ha='right')
            ax.set_yticks(range(0, N_LAYERS, 2))
            ax.set_yticklabels(range(0, N_LAYERS, 2), fontsize=FS_TICK)
        ax0.set_ylabel('Layer', fontsize=FS_LABEL)
        return im0, im2

    for domain in BIAS_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), constrained_layout=True)
        fig.suptitle(
            f'{domain.capitalize()} — NIE heatmap  |  OLMo vs Pythia (cross-architecture)\n'
            'Red = positive NIE (causal signal), blue = negative. '
            'OLMo/Pythia clamped [−1, 1]. Difference: red = Pythia stronger signal.',
            fontsize=FS_SUPTITLE, fontweight='bold')
        result = _draw_row(axes, domain)
        if result:
            im0, im2 = result
            fig.colorbar(im0, ax=[axes[0], axes[1]], label='NIE (clamped [−1, 1])', shrink=0.8)
            fig.colorbar(im2, ax=axes[2],            label='ΔNIE (Pythia − OLMo)',  shrink=0.8)
        _savepdf(fig, os.path.join(out_dir, f'{domain}-olmo-pythia-nie-heatmap.pdf'))

    active = ['gender', 'profession', 'race']
    nr = len(active)
    fig, axes = plt.subplots(nr, 3, figsize=(12, 5 * nr), constrained_layout=True)
    fig.suptitle(
        'NIE heatmap — cross-architecture  |  OLMo (main) vs Pythia (step143k)\n'
        'Red = positive NIE. OLMo/Pythia clamped [−1, 1]. Difference per-domain scaled.',
        fontsize=FS_SUPTITLE, fontweight='bold')
    for row, domain in enumerate(active):
        result = _draw_row(axes[row], domain)
        if result:
            im0, im2 = result
            fig.colorbar(im0, ax=[axes[row, 0], axes[row, 1]], label='NIE',  shrink=0.8)
            fig.colorbar(im2, ax=axes[row, 2],                 label='ΔNIE', shrink=0.8)
    _savepdf(fig, os.path.join(out_dir, 'olmo-pythia-nie-heatmap-composite.pdf'))


def save_olmo_pythia_alp_heatmap(base_stats, pythia_stats, out_dir):
    """
    AbsLogProb heatmap: OLMo (main) | Pythia (step143k) | Difference
    Same layout as save_alp_heatmap.
    Outputs: {domain}-olmo-pythia-alp-heatmap.pdf, olmo-pythia-alp-heatmap-composite.pdf
    """
    base_last   = base_stats[-1]
    pythia_last = _get_pythia(pythia_stats)
    POS_LABELS  = ['[subject]', '[pre-target]', '[target]']
    N_LAYERS    = 16

    def _mat(s):
        return np.column_stack([
            np.array(s['states_score']),
            np.array(s['pre_blank_score']),
            np.array(s['blank_score']),
        ])

    def _draw_row(row_axes, domain):
        ax0, ax1, ax2 = row_axes
        sb = base_last['domains'].get(domain)
        sp = pythia_last['domains'].get(domain)
        if not sb or not sp:
            for ax in [ax0, ax1, ax2]:
                ax.set_visible(False)
            return None
        mb   = _mat(sb)
        mp   = _mat(sp)
        md   = mp - mb
        vmax = max(float(np.max(mb)), float(np.max(mp)))
        vmin = min(0.0, float(np.min(mb)), float(np.min(mp)))
        dlim = max(0.01, float(np.max(np.abs(md))))
        im0 = ax0.imshow(mb, aspect='auto', cmap='YlOrRd', origin='upper', vmin=vmin, vmax=vmax)
        ax1.imshow(mp,       aspect='auto', cmap='YlOrRd', origin='upper', vmin=vmin, vmax=vmax)
        im2 = ax2.imshow(md, aspect='auto', cmap='RdBu_r', origin='upper', vmin=-dlim, vmax=dlim)
        for ax, ttl in zip([ax0, ax1, ax2],
                           ['OLMo (main)', 'Pythia (step143k)', 'Pythia − OLMo']):
            ax.set_title(f'{domain.capitalize()} — {ttl}', fontsize=FS_TITLE)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(POS_LABELS, fontsize=FS_TICK, rotation=15, ha='right')
            ax.set_yticks(range(0, N_LAYERS, 2))
            ax.set_yticklabels(range(0, N_LAYERS, 2), fontsize=FS_TICK)
        ax0.set_ylabel('Layer', fontsize=FS_LABEL)
        return im0, im2

    for domain in BIAS_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), constrained_layout=True)
        fig.suptitle(
            f'{domain.capitalize()} — AbsLogProb heatmap  |  OLMo vs Pythia (cross-architecture)\n'
            'Abs. log prob diff (stereo − anti) at each (position, layer). '
            'YlOrRd: yellow=low, red=high. Difference: red = Pythia stronger signal.',
            fontsize=FS_SUPTITLE, fontweight='bold')
        result = _draw_row(axes, domain)
        if result:
            im0, im2 = result
            fig.colorbar(im0, ax=[axes[0], axes[1]], label='Abs. log prob diff (stereo − anti)', shrink=0.8)
            fig.colorbar(im2, ax=axes[2],            label='Δ Abs. log prob diff (Pythia − OLMo)', shrink=0.8)
        _savepdf(fig, os.path.join(out_dir, f'{domain}-olmo-pythia-alp-heatmap.pdf'))

    active = ['gender', 'profession', 'race']
    nr = len(active)
    fig, axes = plt.subplots(nr, 3, figsize=(12, 5 * nr), constrained_layout=True)
    fig.suptitle(
        'AbsLogProb heatmap — cross-architecture  |  OLMo (main) vs Pythia (step143k)\n'
        'Raw patched scores. YlOrRd for main panels. Difference in RdBu_r (red = Pythia stronger).',
        fontsize=FS_SUPTITLE, fontweight='bold')
    for row, domain in enumerate(active):
        result = _draw_row(axes[row], domain)
        if result:
            im0, im2 = result
            fig.colorbar(im0, ax=[axes[row, 0], axes[row, 1]], label='Abs. log prob diff', shrink=0.8)
            fig.colorbar(im2, ax=axes[row, 2],                 label='Δ Abs. log prob diff', shrink=0.8)
    _savepdf(fig, os.path.join(out_dir, 'olmo-pythia-alp-heatmap-composite.pdf'))


print('\n=== OLMo main vs Pythia step143k: pre-training comparison ===')
save_olmo_vs_pythia_prepost(base_stats, pythia_stats, COMPARE_DIR)

print('\n=== OLMo vs Pythia: NIE heatmap ===')
save_olmo_pythia_nie_heatmap(base_stats, pythia_stats, COMPARE_DIR)

print('\n=== OLMo vs Pythia: AbsLogProb heatmap ===')
save_olmo_pythia_alp_heatmap(base_stats, pythia_stats, COMPARE_DIR)

print('\nDone.')
