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
#
# Runs exist for several model families. scripts/cross_patch.sh writes all of
# them below the repository-local results tree, so plotting reads the same root.

CROSS_PATCH_BASE       = f'{LOCAL_BASE}/cross_patch'
CROSS_PATCH_LOCAL_BASE = CROSS_PATCH_BASE

CROSS_PATCH_FAMILIES = {
    'olmo_1b': {
        'base': CROSS_PATCH_BASE,
        'label': 'OLMo-2 1B',
        'base_model': 'allenai/OLMo-2-0425-1B',
        'instruct_model': 'allenai/OLMo-2-0425-1B-Instruct',
    },
    'qwen2.5_1.5b': {
        'base': CROSS_PATCH_LOCAL_BASE,
        'label': 'Qwen2.5 1.5B',
        'base_model': 'Qwen/Qwen2.5-1.5B',
        'instruct_model': 'Qwen/Qwen2.5-1.5B-Instruct',
    },
    'llama3.2_1b': {
        'base': CROSS_PATCH_LOCAL_BASE,
        'label': 'Llama-3.2 1B',
        'base_model': 'meta-llama/Llama-3.2-1B',
        'instruct_model': 'meta-llama/Llama-3.2-1B-Instruct',
    },
    'gemma3_1b': {
        'base': CROSS_PATCH_LOCAL_BASE,
        'label': 'Gemma-3 1B',
        'base_model': 'google/gemma-3-1b-pt',
        'instruct_model': 'google/gemma-3-1b-it',
    },
}

# 'dir' is the olmo_1b run directory — kept for existing scripts (table.py,
# regenerate_compare_plots.py) that predate the multi-family layout.
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


def cross_patch_cases_dir(family, direction_key, domain):
    fam = CROSS_PATCH_FAMILIES[family]
    return os.path.join(fam['base'], f'{family}_{direction_key}',
                        domain, 'causal_trace', 'cases')

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

# Display heuristic (not a statistical test): panels whose effect_gap falls below
# this get a ⚠ marker, because NIE divides by the gap and small gaps blow the
# ratio far outside its interpretable [0, 1] range. Measured over the 68
# (model, checkpoint, domain) entries in plots/*/stats.json: gap ≈ 0.12 → max|NIE|
# 0.24–0.56, whereas gap = 0.0056 → 9.98 and gap = 0.0142 → 4.12. The cut is a
# round number, not a derived bound — it flags 7 of the 9 entries with
# max|NIE| > 1.5 and also flags 8 well-behaved ones. Nothing is excluded from any
# computation; see normalized_indirect_effect() for the degenerate-gap policy.
LOW_SIGNAL = 0.03

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


# ── normalized indirect effect ────────────────────────────────────────────────

def normalized_indirect_effect(values, mean_low, effect_gap, degenerate='skip'):
    """Normalized Indirect Effect — the single definition used by every plot.

        NIE = (restoration_score − low_score) / (high_score − low_score)

    with effect_gap = high_score − low_score, so NIE = 1 is full recovery of the
    clean-vs-corrupted signal and NIE = 0 is none. Convention follows Sen Sharma
    et al. (COLM 2024) Eq. 7 and Zhang & Nanda (ICLR 2024); tracing protocol from
    Meng et al. (2022).

    NIE is undefined when effect_gap <= 0 — the denominator vanishes, or goes
    negative (corruption *raised* the score), which silently inverts the scale.
    Callers pick that behaviour explicitly rather than inheriting a default:

        'skip' → None                    caller drops the series
        'zero' → 0.0 / zeros(shape)      renders as "no effect"
        'nan'  → nan / full(shape, nan)  renders as a gap

    A non-finite gap also takes the degenerate branch. Scalars return floats;
    lists and arrays return an ndarray of the same shape. The input dtype is
    preserved — the cached score arrays are float32, and promoting them to
    float64 here would shift every plotted value by ~1e-7.
    """
    scalar = np.ndim(values) == 0
    if effect_gap > 0:
        out = (np.asarray(values) - mean_low) / effect_gap
        return float(out) if scalar else out
    if degenerate == 'skip':
        return None
    fill = 0.0 if degenerate == 'zero' else float('nan')
    return fill if scalar else np.full(np.shape(values), fill, dtype=float)

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
    """Return the same case subset for full, attention, and MLP restores."""
    single_by_case = {_case_stem(name): name for name in single}
    attn_by_case = {_case_stem(name): name for name in attn}
    mlp_by_case = {_case_stem(name): name for name in mlp}
    common = set(single_by_case) & set(attn_by_case) & set(mlp_by_case)
    chosen = [_case_stem(name) for name in single if _case_stem(name) in common]
    if num_sample is not None:
        chosen = chosen[:num_sample]
    return (
        [single_by_case[case] for case in chosen],
        [attn_by_case[case] for case in chosen],
        [mlp_by_case[case] for case in chosen],
    )


def load_npz_local(path):
    return np.load(path, allow_pickle=True)


def load_npz_zip(zf, zip_path):
    with zf.open(zip_path) as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=True)


SCORE_METRIC = 'sentence_abs_logprob_diff_v1'
STATS_SCORE_FIELDS = {
    'states_nie': 'states_score',
    'attn_nie': 'attn_score',
    'mlp_nie': 'mlp_score',
    'pre_blank_nie': 'pre_blank_score',
    'blank_nie': 'blank_score',
}


def normalize_stats_score_fields(domain_stats):
    """Expose correctly named raw-score fields when reading legacy stats.json."""
    for legacy_key, score_key in STATS_SCORE_FIELDS.items():
        if score_key not in domain_stats and legacy_key in domain_stats:
            domain_stats[score_key] = domain_stats[legacy_key]
    return domain_stats


def _metadata_value(data, key):
    if key not in data:
        return None
    return np.asarray(data[key]).item()


def result_metadata(data):
    """Return the fields that define whether two result files are compatible."""
    scores = np.asarray(data['scores'])
    if scores.ndim < 1:
        raise ValueError('scores must have a layer dimension')
    score_layers = int(scores.shape[-1])
    declared_layers = _metadata_value(data, 'num_layers')
    if declared_layers is not None and int(declared_layers) != score_layers:
        raise ValueError(
            f'num_layers={declared_layers} but scores has {score_layers} layers'
        )
    return {
        'score_metric': _metadata_value(data, 'score_metric'),
        'source_model': _metadata_value(data, 'source_model'),
        'target_model': _metadata_value(data, 'target_model'),
        'source_revision': _metadata_value(data, 'source_revision'),
        'target_revision': _metadata_value(data, 'target_revision'),
        'direction': _metadata_value(data, 'direction'),
        'num_layers': score_layers,
    }


def validate_score_files(file_lists, loader, expected=None):
    """Reject mixed or incorrectly labelled NPZ files before aggregation.

    A homogeneous legacy set without metadata is still readable, but it is
    reported explicitly. Mixing legacy and current-schema files is rejected.
    """
    reference = None
    for item in (item for file_list in file_lists for item in file_list):
        try:
            metadata = result_metadata(loader(item))
        except Exception as exc:
            raise ValueError(f'cannot validate {item}: {exc}') from exc

        if metadata['score_metric'] is not None:
            required = (
                'source_model', 'target_model',
                'source_revision', 'target_revision', 'direction',
            )
            missing = [key for key in required if metadata[key] is None]
            if missing:
                raise ValueError(
                    f'{item}: current-schema result is missing metadata {missing}'
                )

        if expected is not None:
            for key, expected_value in expected.items():
                actual_value = metadata.get(key)
                if actual_value is not None and actual_value != expected_value:
                    raise ValueError(
                        f'{item}: expected {key}={expected_value!r}, '
                        f'found {actual_value!r}'
                    )

        if reference is None:
            reference = metadata
        elif metadata != reference:
            raise ValueError(
                f'{item}: result metadata differs from other files in this group'
            )

    if reference is not None and reference['score_metric'] is None:
        print('    [metadata] homogeneous legacy NPZ set; direction is inferred from its directory')
    return reference


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
        except Exception as exc:
            print(f'    [collect_scores] Skipping {item}: {exc}')
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
