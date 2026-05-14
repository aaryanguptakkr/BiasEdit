"""
Shared constants and helpers used across bias-tracing plot scripts.

Can be imported from the bias_tracing root (fig.py) or from scripts/:
    from plot_utils import BAR_COLORS, collect_scores, _draw_bars, ...

Scripts in scripts/ need to add the parent directory to sys.path:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""

import os
import io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────

ZIP_PATH   = '/deepfreeze/share/xuxin_transfer/bias_tracing/results.zip'
LOCAL_BASE = '/deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing/results'
PLOTS_BASE = '/deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing/plots'

# ── model references ──────────────────────────────────────────────────────────

BASE_MODEL = 'OLMo-2-0425-1B'
INST_MODEL = 'OLMo-2-0425-1B-Instruct'
PYTHIA     = 'pythia-1b'

# ── model / checkpoint catalogue ─────────────────────────────────────────────

MODEL_CONFIGS = {
    'OLMo-2-0425-1B': {
        'org': 'allenai',
        'checkpoints': [
            ('stage1-step0-tokens0B',                  '0B'),
            ('stage1-step10000-tokens21B',              '21B'),
            ('stage1-step150000-tokens315B',            '315B'),
            ('stage1-step1140000-tokens2391B',          '2.4T'),
            ('stage1-step1907359-tokens4001B',          '4T'),
            ('stage2-ingredient3-step1000-tokens3B',    's2-3B'),
            ('stage2-ingredient3-step11000-tokens24B',  's2-24B'),
            ('stage2-ingredient3-step23852-tokens51B',  's2-51B'),
        ],
    },
    'OLMo-2-0425-1B-Instruct': {
        'org': 'allenai',
        'checkpoints': [
            ('step_200',  'step200'),
            ('step_1400', 'step1400'),
            ('step_2600', 'step2600'),
        ],
    },
    'pythia-1b': {
        'org': 'EleutherAI',
        'checkpoints': [
            ('step0',      'step0'),
            ('step1000',   'step1k'),
            ('step5000',   'step5k'),
            ('step81000',  'step81k'),
            ('step137000', 'step137k'),
            ('step143000', 'step143k'),
        ],
    },
}

# ── cross-patch configuration ─────────────────────────────────────────────────
#
# Cross-patching injects hidden states from a *source* model into a *target*
# model at one (token, layer) position at a time, then measures how much the
# prediction recovers toward the source model's bias-consistent output.
#
# Two directions capture the asymmetry between pre-trained and instruct models:
#   pre_to_post — source: base pre-trained  /  target: instruct fine-tuned
#   post_to_pre — source: instruct fine-tuned  /  target: base pre-trained
#
# The .npz format is identical to within-model causal tracing, so the same
# collect_scores / _draw_bars pipeline applies.

CROSS_PATCH_BASE = '/deepfreeze/share/omar_transfer/results/cross_patch'

CROSS_PATCH_CONFIGS = {
    'pre_to_post': {
        'dir':   'olmo_1b_pre_to_post',
        'label': 'Pre → Post',
        'desc':  'Base activations patched into instruct model',
    },
    'post_to_pre': {
        'dir':   'olmo_1b_post_to_pre',
        'label': 'Post → Pre',
        'desc':  'Instruct activations patched into base model',
    },
}

# ── style constants ───────────────────────────────────────────────────────────

# Font sizes
FS_SUPTITLE = 11   # figure-level title (suptitle)
FS_TITLE    = 10   # panel / subplot title
FS_LABEL    =  9   # axis labels (xlabel, ylabel)
FS_TICK     =  8   # tick labels and x-tick rotation labels
FS_LEGEND   =  7   # legend text
FS_ANNOT    =  7   # small in-plot annotations (⚠ text, footnotes)

# Figure dimensions (inches)
FIG_BAR_W_SINGLE   = 12.0   # width of a standalone single-domain bar chart
FIG_BAR_H_SINGLE   =  5.0   # height of a standalone single-domain bar chart
FIG_BAR_W_PER_COL  =  3.5   # per-column width in wide domain-composite figures (1 row × N domains)
FIG_GRID_W_PER_COL =  7.0   # per-column width in 2-col checkpoint-grid figures
FIG_ROW_H          =  4.0   # per-row height for all bar/line figures
FIG_LINE_W_PER_PAN =  5.0   # per-panel width in 3-panel line plots
FIG_TRAJ_H         = 10.0   # trajectory plot height (width is dynamic)

# Colors — training phase
BASE_COLOR     = '#1565C0'  # base pre-training (trajectory / comparison line plots)
INSTRUCT_COLOR = '#E65100'  # instruct fine-tuning
PYTHIA_COLOR   = '#2E7D32'  # Pythia checkpoints in cross-architecture plots

# Colors — annotations
LOW_SIG_COLOR  = '#B71C1C'  # ⚠ annotation and low-signal text
LOW_SIG_BG     = '#FFF9C4'  # yellow subplot background for low-signal checkpoints

# Bar width (fraction of layer spacing used per bar)
BAR_WIDTH = 0.25

# ── domain / plot constants ───────────────────────────────────────────────────

BIAS_TYPES = ['gender', 'profession', 'race', 'religion']

LOW_SIGNAL = 0.03  # effect_gap below this → ⚠ marker; tracing estimates unreliable

STATES_LABELS = [
    'Effect of single state',
    'Effect with Attn severed',
    'Effect with MLP severed',
]
WORDS_LABELS = [
    'Effect of bias attribute words',
    'Effect of the token before attribute terms',
    'Effect of attribute terms',
]

# Standardized bar colors: blue (full restore), red (Attn severed / MLP-only), green (MLP severed / Attn-only)
BAR_COLORS = ['#1f77b4', '#d62728', '#2ca02c']

Y_LABEL_BARS = 'Abs. log prob diff (stereo − anti)'

# ── data path helpers ─────────────────────────────────────────────────────────

def local_cases_dir(model_name, org, checkpoint, domain):
    return os.path.join(LOCAL_BASE, org, model_name,
                        checkpoint, domain, 'causal_trace', 'cases')


def zip_cases_prefix(org, model_name, checkpoint, domain):
    return f'results/{org}/{model_name}/{checkpoint}/{domain}/causal_trace/cases/'


def partition_names(names):
    """Split filenames into (full-restore, attn-only, mlp-only) lists."""
    single, attn, mlp = [], [], []
    for n in names:
        if '_attn.' in n:
            attn.append(n)
        elif '_mlp.' in n or '_intermediate.' in n:
            mlp.append(n)
        elif n.endswith('.npz'):
            single.append(n)
    return single, attn, mlp


def load_npz_local(path):
    return np.load(path, allow_pickle=True)


def load_npz_zip(zf, zip_path):
    with zf.open(zip_path) as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=True)


def collect_scores(file_list, loader):
    """
    Aggregate indirect-effect scores across sentence pairs.

    Returns:
      bias_mean      (n_layers,)  mean score at subject token positions
      pre_blank_mean (n_layers,)  mean score at token before prediction target
      blank_mean     (n_layers,)  mean score at prediction target positions
      n_cases        int
      mean_high      float  mean clean-run abs log prob diff
      mean_low       float  mean corrupted-run abs log prob diff
    Returns (None, None, None, 0, 0.0, 0.0) on failure.
    """
    bias_word, pre_blank, blank = [], [], []
    highs, lows = [], []
    for item in tqdm(file_list, leave=False):
        try:
            d = loader(item)
            scores = d['scores']
            for b, e in d['corrupt_range_anti']:
                bias_word.append(scores[b:e])
            idx0 = int(d['blank_idxs_anti'][0])
            idx1 = int(d['blank_idxs_anti'][1]) if len(d['blank_idxs_anti']) > 1 else idx0 + 1
            if idx0 > 0:
                pre_blank.append(scores[idx0 - 1][np.newaxis, :])
            blank.append(scores[idx0:idx1])
            highs.append(float(d['high_score']))
            lows.append(float(d['low_score']))
        except Exception:
            continue
    if not bias_word:
        return None, None, None, 0, 0.0, 0.0
    n_layers = bias_word[0].shape[-1]
    return (
        np.mean(np.concatenate(bias_word, axis=0), axis=0),
        np.mean(np.concatenate(pre_blank, axis=0), axis=0) if pre_blank else np.zeros(n_layers),
        np.mean(np.concatenate(blank,     axis=0), axis=0),
        len(highs),
        float(np.mean(highs)),
        float(np.mean(lows)),
    )

# ── plotting helpers ──────────────────────────────────────────────────────────

def _draw_bars(ax, r1, r2, r3, labels, colors, num_layer, xlabel, ylabel, title):
    """
    Draw a 3-bar-per-layer bar chart. Auto-sets ylim from data; callers
    can call ax.set_ylim() afterwards to override (e.g. for fixed axes).
    """
    xs = np.arange(len(r1))
    ax.bar(xs,                 r1, color=colors[0], width=BAR_WIDTH, edgecolor='gray', label=labels[0])
    ax.bar(xs + BAR_WIDTH,     r2, color=colors[1], width=BAR_WIDTH, edgecolor='gray', label=labels[1])
    ax.bar(xs + 2 * BAR_WIDTH, r3, color=colors[2], width=BAR_WIDTH, edgecolor='gray', label=labels[2])
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=FS_LABEL)
    ax.set_xticks(np.arange(0, num_layer, max(1, num_layer // 8)))
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND)
    all_vals = np.concatenate([r1, r2, r3])
    margin = (all_vals.max() - all_vals.min()) * 0.1 or 0.05
    ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)


def _savepdf(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {path}')
