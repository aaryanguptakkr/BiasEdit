"""
Generate checkpoint × layer heatmaps for all three model families.

For each model a PNG is saved at:
  results/{org}/{model}/heatmap_checkpoint_layer.pdf

Layout: 2 rows (MLP, Attn) × 4 columns (gender, profession, race, religion)
X-axis: training checkpoints (ordered by training progression)
Y-axis: transformer layers
Colour:  mean peak-window indirect effect score

Reads .npz case files directly from the shared zip, falling back to local
filesystem for checkpoints that are already extracted.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_utils import (
    ZIP_PATH, LOCAL_BASE, PLOTS_BASE, MODEL_CONFIGS, BIAS_TYPES,
    FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK,
)

DOMAINS    = BIAS_TYPES
KINDS      = ['mlp', 'attn']
NUM_LAYERS = 16

# ── helpers ───────────────────────────────────────────────────────────────────

def _subject_mean(d):
    """Mean indirect effect over subject token positions (corrupt_range_anti) per layer."""
    scores = d['scores'].astype(np.float32)  # (n_tokens, n_layers)
    rows = [scores[b:e] for b, e in d['corrupt_range_anti']]
    return np.concatenate(rows, axis=0).mean(axis=0)  # (n_layers,)


def load_mean_scores_local(cases_dir, kind):
    """Read from extracted filesystem directory."""
    files = [f for f in os.listdir(cases_dir) if f.endswith(f'_{kind}.npz')]
    if not files:
        return None
    per_case = []
    for fname in files:
        try:
            d = np.load(os.path.join(cases_dir, fname), allow_pickle=True)
            per_case.append(_subject_mean(d))
        except Exception:
            continue
    return np.stack(per_case).mean(axis=0) if per_case else None


_zip_names = None  # cached after first open

def load_mean_scores_zip(zf, zip_prefix, kind):
    """Read from zip file."""
    global _zip_names
    if _zip_names is None:
        _zip_names = zf.namelist()
    suffix = f'_{kind}.npz'
    matching = [n for n in _zip_names
                if n.startswith(zip_prefix) and n.endswith(suffix)]
    if not matching:
        return None
    per_case = []
    for name in matching:
        try:
            with zf.open(name) as f:
                d = np.load(io.BytesIO(f.read()), allow_pickle=True)
            per_case.append(_subject_mean(d))
        except Exception:
            continue
    return np.stack(per_case).mean(axis=0) if per_case else None


def load_mean_scores(zf, org, model_name, checkpoint, domain, kind, source='zip'):
    """
    source='zip'   — always read from zip (fast, recommended on deepfreeze)
    source='local' — always read from extracted NFS files
    source='auto'  — prefer local if extracted, else zip
    """
    local_dir = os.path.join(LOCAL_BASE, org, model_name,
                             checkpoint, domain, 'causal_trace', 'cases')
    use_local = (source == 'local') or (source == 'auto' and os.path.isdir(local_dir))

    if use_local:
        return load_mean_scores_local(local_dir, kind)

    prefix = f'results/{org}/{model_name}/{checkpoint}/{domain}/causal_trace/cases/'
    return load_mean_scores_zip(zf, prefix, kind)


# ── main ──────────────────────────────────────────────────────────────────────

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='zip', choices=['zip', 'local', 'auto'],
                    help='zip: read from zip (~fast); local: read from NFS files (~slow); '
                         'auto: local if extracted else zip (default: zip)')
heatmap_args = parser.parse_args()

print(f'Opening zip: {ZIP_PATH}')
zf = zipfile.ZipFile(ZIP_PATH, 'r')

for model_name, cfg in MODEL_CONFIGS.items():
    org = cfg['org']
    checkpoints = cfg['checkpoints']
    step_names  = [c[0] for c in checkpoints]
    step_labels = [c[1] for c in checkpoints]
    n_steps     = len(step_names)

    print(f'\n=== {model_name} ({n_steps} checkpoints) ===')

    # ── build (domain, kind) → (n_layers × n_steps) matrix ──────────────────
    matrices = {}
    for domain in DOMAINS:
        for kind in KINDS:
            mat = np.full((NUM_LAYERS, n_steps), np.nan)
            for si, step in enumerate(step_names):
                scores = load_mean_scores(zf, org, model_name, step, domain, kind,
                                         source=heatmap_args.source)
                if scores is not None:
                    mat[:, si] = scores[:NUM_LAYERS]
                    print(f'  {domain}/{kind}/{step}: loaded (peak layer {int(np.argmax(scores))})')
                else:
                    print(f'  {domain}/{kind}/{step}: no data')
            matrices[(domain, kind)] = mat

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        len(KINDS), len(DOMAINS),
        figsize=(max(10, n_steps * 1.4 * len(DOMAINS)), 9),
        constrained_layout=True,
    )
    if len(KINDS) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'{model_name}  |  Causal Tracing — Mean Indirect Effect\n'
        'X: Training Checkpoint   Y: Layer   Colour: peak-window mean score',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    CMAP = 'hot'

    for ki, kind in enumerate(KINDS):
        for di, domain in enumerate(DOMAINS):
            ax = axes[ki, di]
            mat = matrices[(domain, kind)]

            vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0
            vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1
            im = ax.imshow(
                mat, aspect='auto', cmap=CMAP, origin='lower',
                vmin=vmin, vmax=vmax, interpolation='nearest',
            )

            ax.set_xticks(range(n_steps))
            ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=FS_TICK)
            ax.set_yticks(range(0, NUM_LAYERS, 2))
            ax.set_yticklabels(range(0, NUM_LAYERS, 2), fontsize=FS_TICK)

            if ki == 0:
                ax.set_title(domain.capitalize(), fontsize=FS_TITLE,
                             fontweight='bold', pad=6)
            if di == 0:
                ax.set_ylabel(f'{kind.upper()}\nLayer', fontsize=FS_LABEL)
            if ki == len(KINDS) - 1:
                ax.set_xlabel('Checkpoint', fontsize=FS_LABEL)

            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=FS_TICK)

    out_path = os.path.join(PLOTS_BASE, model_name, 'heatmap_checkpoint_layer.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out_path}')

zf.close()
print('\nDone.')
