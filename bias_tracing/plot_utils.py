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

try:
    from private_names import name1, name2, name3, name4
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "Missing private_names.py (gitignored). Create it next to plot_utils.py "
        "defining name1..name4 — see private_names.py for the expected fields."
    ) from _e

# ── paths ─────────────────────────────────────────────────────────────────────

ZIP_PATH   = f'/deepfreeze/share/{name3}/bias_tracing/results.zip'
MAIN_ZIP   = f'/deepfreeze/share/{name3}/bias_tracing/main.zip'
LOCAL_BASE = f'/deepfreeze/{name1}/{name2}/BiasEdit/bias_tracing/results'
PLOTS_BASE = f'/deepfreeze/{name1}/{name2}/BiasEdit/bias_tracing/plots'

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
            ('main',                                    'main'),
        ],
    },
    'OLMo-2-0425-1B-Instruct': {
        'org': 'allenai',
        'checkpoints': [
            ('step_200',  'step200'),
            ('step_1400', 'step1400'),
            ('step_2000', 'step2000'),
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

CROSS_PATCH_BASE = f'/deepfreeze/share/{name4}/results/cross_patch'

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

BIAS_TYPES    = ['gender', 'profession', 'race', 'religion']
PAPER_DOMAINS = ['gender', 'race', 'profession']  # domains used in paper figures (no religion)

LOW_SIGNAL = 0.03  # effect_gap below this → ⚠ marker; tracing estimates unreliable

STATES_LABELS = [
    'Effect of single state',
    'Effect with Attn severed',
    'Effect with MLP severed',
]
WORDS_LABELS = [
    'Effect of subject token',     # bias_mean       — subject (bias attribute) token positions
    'Effect of pre-target token',  # pre_blank_mean  — token before the prediction target
    'Effect of target token',      # blank_mean      — prediction target token positions
]

# Standardized bar colors: blue (full restore), red (Attn severed / MLP-only), green (MLP severed / Attn-only)
BAR_COLORS    = ['#1f77b4', '#d62728', '#2ca02c']  # states bars
STATES_COLORS = BAR_COLORS                          # alias used in appendix grids
WORDS_COLORS  = ['#9467bd', '#ff7f0e', '#17becf']  # distinct palette for words bars

Y_LABEL_BARS = 'Abs. log prob diff (stereo − anti)'
Y_LABEL_NIE  = 'NIE (normalized indirect effect)'

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


def _case_stem(name):
    """Case id shared by a case's full/attn/mlp files.
    'knowledge_7.npz' / 'knowledge_7_attn.npz' / 'knowledge_7_mlp.npz' → 'knowledge_7'."""
    s = name[:-4] if name.endswith('.npz') else name
    for suf in ('_attn', '_mlp', '_intermediate'):
        if s.endswith(suf):
            return s[:-len(suf)]
    return s


def subsample_aligned(single, attn, mlp, num_sample):
    """Subsample the three restore-type file lists so they cover the SAME cases.

    Slicing single/attn/mlp independently with [:num_sample] can pick mismatched
    case subsets (the lists are sorted lexicographically and a case may be missing
    one of its three files), which would average the three bars over different
    cases. This takes the first num_sample full-restore cases, then keeps only the
    attn/mlp files belonging to those same cases.

    num_sample=None → return all three unchanged (the all-data path).
    """
    if num_sample is None:
        return single, attn, mlp
    chosen = single[:num_sample]
    keep   = {_case_stem(s) for s in chosen}
    return (chosen,
            [a for a in attn if _case_stem(a) in keep],
            [m for m in mlp  if _case_stem(m) in keep])


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

def _draw_bars(ax, r1, r2, r3, labels, colors, num_layer, xlabel, ylabel, title,
               fs_label=None, fs_tick=None, fs_title=None):
    """
    Draw a 3-bar-per-layer bar chart. Auto-sets ylim from data; callers
    can call ax.set_ylim() afterwards to override (e.g. for fixed axes).
    fs_label/fs_tick override FS_LABEL/FS_TICK when provided; title always uses FS_TITLE.
    """
    _fl = fs_label if fs_label is not None else FS_LABEL
    _ft = fs_title if fs_title is not None else FS_TITLE
    xs = np.arange(len(r1))
    ax.bar(xs,                 r1, color=colors[0], width=BAR_WIDTH, edgecolor='gray', label=labels[0])
    ax.bar(xs + BAR_WIDTH,     r2, color=colors[1], width=BAR_WIDTH, edgecolor='gray', label=labels[1])
    ax.bar(xs + 2 * BAR_WIDTH, r3, color=colors[2], width=BAR_WIDTH, edgecolor='gray', label=labels[2])
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=_fl)
    ax.set_xticks(np.arange(0, num_layer, max(1, num_layer // 8)))
    ax.set_ylabel(ylabel, fontsize=_fl)
    ax.set_title(title, fontsize=_ft)
    ax.legend(fontsize=FS_LEGEND)
    if fs_tick is not None:
        ax.tick_params(labelsize=fs_tick)
    all_vals = np.concatenate([r1, r2, r3])
    margin = (all_vals.max() - all_vals.min()) * 0.1 or 0.05
    ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)


def _savepdf(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {path}')
