"""
Generate bar-chart bias-tracing plots for every model × checkpoint × domain.

Per checkpoint  →  plots/{model}/{label}/
    {domain}-states.pdf         single/attn-severed/mlp-severed bars per layer
    {domain}-words.pdf          bias-word/pre-blank/blank-token bars per layer
    composite-states.pdf        all 4 domains side-by-side (states)
    composite-words.pdf         all 4 domains side-by-side (words)
    composite-all.pdf           2×4 grid: top=states, bottom=words

Per domain (across all checkpoints)  →  plots/{model}/
    {domain}-states-all-checkpoints.pdf   one subplot per checkpoint
    {domain}-words-all-checkpoints.pdf    one subplot per checkpoint

Per model  →  plots/{model}/
    stats.json    all numeric data (NIE by layer, peak layers, n_cases, scores)
    report.md     human-readable summary tables

Data is read directly from the shared zip; falls back to local filesystem
for checkpoints that are already extracted.
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

# ── paths ────────────────────────────────────────────────────────────────────

ZIP_PATH   = '/deepfreeze/share/xuxin_transfer/bias_tracing/results.zip'
LOCAL_BASE = '/deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing/results'
PLOTS_BASE = '/deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing/plots'

# ── model / checkpoint catalogue ─────────────────────────────────────────────

MODEL_CONFIGS = {
    'OLMo-2-0425-1B': {
        'zip_org':   'allenai',
        'local_org': 'allenai',
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
        'zip_org':   'allenai',
        'local_org': 'allenai',
        'checkpoints': [
            ('step_200',  'step200'),
            ('step_1400', 'step1400'),
            ('step_2600', 'step2600'),
        ],
    },
    'pythia-1b': {
        'zip_org':   'EleutherAI',
        'local_org': 'EleutherAI',
        'checkpoints': [
            ('step0',     'step0'),
            ('step1000',  'step1k'),
            ('step5000',  'step5k'),
            ('step81000', 'step81k'),
            ('step137000','step137k'),
            ('step143000','step143k'),
        ],
    },
}

BIAS_TYPES    = ['gender', 'profession', 'race', 'religion']
STATES_LABELS = ['Effect of single state',
                 'Effect with Attn severed',
                 'Effect with MLP severed']
WORDS_LABELS  = ['Effect of bias attribute words',
                 'Effect of the token before attribute terms',
                 'Effect of attribute terms']
BAR_COLORS    = ['blue', 'red', 'green']

# ── CLI ───────────────────────────────────────────────────────────────────────

PLOT_CHOICES = ['bars', 'delta', 'compare']

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=__doc__,
    epilog="""
examples:
  python fig.py                              # everything (default)
  python fig.py --plots delta compare        # only delta + base-vs-instruct, skip bar charts
  python fig.py --plots bars                 # only per-checkpoint bar charts
  python fig.py --model pythia-1b --plots bars delta
""")
parser.add_argument('--model', default=None, choices=list(MODEL_CONFIGS.keys()),
                    help='run only this model; omit to run all')
parser.add_argument('--bias', default=None, choices=BIAS_TYPES,
                    help='run only this domain; omit to run all four')
parser.add_argument('--num_sample', type=int, default=None,
                    help='max cases per domain per kind (default: all)')
parser.add_argument('--source', default='zip', choices=['zip', 'local', 'auto'],
                    help='zip: always read from zip (recommended); '
                         'local: read from extracted NFS files; '
                         'auto: prefer local if extracted, else zip')
parser.add_argument('--plots', nargs='+', default=['all'], choices=PLOT_CHOICES + ['all'],
                    metavar='PLOT',
                    help=('which plots to generate: bars, delta, compare, all (default: all). '
                          'bars = per-checkpoint bar charts + composites; '
                          'delta = bias-acquired-per-step figure; '
                          'compare = base vs instruct comparison.'))
args = parser.parse_args()

_plots = set(args.plots)
if 'all' in _plots:
    _plots = set(PLOT_CHOICES)
RUN_BARS    = 'bars'    in _plots
RUN_DELTA   = 'delta'   in _plots
RUN_COMPARE = 'compare' in _plots

models_to_run  = [args.model] if args.model else list(MODEL_CONFIGS.keys())
domains_to_run = [args.bias]  if args.bias  else BIAS_TYPES

# ── data helpers ──────────────────────────────────────────────────────────────

def local_cases_dir(model_name, org, checkpoint, domain):
    return os.path.join(LOCAL_BASE, org, model_name,
                        checkpoint, domain, 'causal_trace', 'cases')

def zip_cases_prefix(org, model_name, checkpoint, domain):
    return f'results/{org}/{model_name}/{checkpoint}/{domain}/causal_trace/cases/'

def partition_names(names):
    single, attn, mlp = [], [], []
    for n in names:
        if '_attn.' in n:
            attn.append(n)
        elif '_mlp.' in n or '_intermediate.' in n:
            mlp.append(n)
        elif n.endswith('.npz'):
            single.append(n)
    return single, attn, mlp

def _load_local(path):
    return np.load(path, allow_pickle=True)

def _load_zip(zf, zip_path):
    with zf.open(zip_path) as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=True)

def collect_scores(file_list, loader):
    """
    Returns:
      bias_mean      (n_layers,)  mean score at subject token positions
      pre_blank_mean (n_layers,)  mean score at token before prediction target
      blank_mean     (n_layers,)  mean score at prediction target positions
      n_cases        int          number of cases successfully loaded
      mean_high      float        mean baseline (uncorrupted) score
      mean_low       float        mean corrupted-only score
    Returns (None,...) on failure.
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
        np.mean(np.concatenate(pre_blank, axis=0), axis=0) if pre_blank
            else np.zeros(n_layers),
        np.mean(np.concatenate(blank,     axis=0), axis=0),
        len(highs),
        float(np.mean(highs)),
        float(np.mean(lows)),
    )

# ── plotting helpers ──────────────────────────────────────────────────────────

def _draw_bars(ax, r1, r2, r3, labels, colors, num_layer, xlabel, ylabel, title):
    bar_width = 0.25
    xs = np.arange(len(r1))
    ax.bar(xs,               r1, color=colors[0], width=bar_width, edgecolor='gray', label=labels[0])
    ax.bar(xs + bar_width,   r2, color=colors[1], width=bar_width, edgecolor='gray', label=labels[1])
    ax.bar(xs + 2*bar_width, r3, color=colors[2], width=bar_width, edgecolor='gray', label=labels[2])
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=8)
    ax.set_xticks(np.arange(0, num_layer, max(1, num_layer // 8)))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6)
    all_vals = np.concatenate([r1, r2, r3])
    margin = (all_vals.max() - all_vals.min()) * 0.1 or 0.05
    ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)

def _savepdf(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {path}')

def save_individual(r1, r2, r3, labels, colors, num_layer, title, savepath):
    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_bars(ax, r1, r2, r3, labels, colors, num_layer,
               'Layer', 'Abs. log prob diff (stereo − anti)', title)
    plt.tight_layout()
    _savepdf(fig, savepath)

def save_composite(domain_data, plot_type, model_name, ckpt_label, out_dir):
    """1×4 grid — all domains side-by-side for one checkpoint."""
    domains = [d for d in BIAS_TYPES if d in domain_data]
    if not domains:
        return
    labels       = STATES_LABELS if plot_type == 'states' else WORDS_LABELS
    title_suffix = 'effect of states' if plot_type == 'states' else 'effect of different words'
    fig, axes = plt.subplots(1, len(domains), figsize=(13, 4))
    if len(domains) == 1:
        axes = [axes]
    fig.suptitle(f'Bias {title_suffix} — {model_name}  [{ckpt_label}]',
                 fontsize=11, fontweight='bold')
    for ax, domain in zip(axes, domains):
        r1, r2, r3, nl = domain_data[domain]
        _draw_bars(ax, r1, r2, r3, labels, BAR_COLORS, nl,
                   'Layer', 'Abs. log prob diff (stereo − anti)', domain.title())
    plt.tight_layout()
    _savepdf(fig, os.path.join(out_dir, f'composite-{plot_type}.pdf'))

def save_composite_all(states_data, words_data, model_name, ckpt_label, out_dir):
    """2×4 grid — top row=states, bottom row=words, columns=domains."""
    domains = [d for d in BIAS_TYPES if d in states_data and d in words_data]
    if not domains:
        return
    fig, axes = plt.subplots(2, len(domains), figsize=(13, 8))
    fig.suptitle(f'All bias domains — {model_name}  [{ckpt_label}]',
                 fontsize=12, fontweight='bold')
    for col, domain in enumerate(domains):
        r1, r2, r3, nl = states_data[domain]
        _draw_bars(axes[0, col], r1, r2, r3, STATES_LABELS, BAR_COLORS, nl,
                   'Layer', 'Abs. log prob diff (stereo − anti)', f'{domain.title()} — states')
        r1, r2, r3, nl = words_data[domain]
        _draw_bars(axes[1, col], r1, r2, r3, WORDS_LABELS, BAR_COLORS, nl,
                   'Layer', 'Abs. log prob diff (stereo − anti)', f'{domain.title()} — words')
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

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]
    fig.suptitle(f'{domain.title()} bias {title_suffix} — {model_name} (all checkpoints)',
                 fontsize=12, fontweight='bold')

    for i, (ckpt_label, r1, r2, r3, nl) in enumerate(domain_arrays):
        _draw_bars(axes_flat[i], r1, r2, r3, labels, BAR_COLORS, nl,
                   'Layer', 'Abs. log prob diff (stereo − anti)', ckpt_label)

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
    LOW_SIGNAL_THRESHOLD = 0.03
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
            flag = ' ⚠' if gap < LOW_SIGNAL_THRESHOLD else ''
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
    LOW_SIGNAL_THRESHOLD = 0.03
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
                flag = '  ⚠ low-signal' if gap < LOW_SIGNAL_THRESHOLD else ''
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

LOW_SIGNAL = 0.03
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
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                                 constrained_layout=True)
        axes_flat = axes.flatten() if n > 1 else [axes]

        fig.suptitle(
            f'{model_name} — {domain.capitalize()} bias: Δ per layer vs. previous checkpoint\n'
            'Bars show change in abs. log prob diff (curr − prev) at each layer. '
            'Positive = more causal effect acquired.  Y-axis fixed across all intervals.',
            fontsize=11, fontweight='bold')

        for ax, (pair_label, ds, da, dm, nl, low_sig) in zip(axes_flat, pairs):
            _draw_bars(ax, ds, da, dm, DELTA_LABELS, BAR_COLORS, nl,
                       'Layer', 'Δ Abs. log prob diff (stereo − anti)', pair_label)
            ax.set_ylim(y_min, y_max)   # override per-subplot limits
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            if low_sig:
                ax.set_facecolor('#FFF9C4')
                ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                        fontsize=7, ha='right', va='top', color='#B71C1C')

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
      3. Raw abs. log prob diff at L0 (states_nie[0]) — absolute causal signal at embedding.
    Base checkpoints appear on the left; instruct fine-tuning on the right;
    a vertical dashed line marks the phase boundary.
    Output: {domain}-bias-trajectory.pdf  (one per domain)
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
    BAR_LABELS = ['Effect of single state', 'Effect with Attn severed', 'Effect with MLP severed']

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
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                                 constrained_layout=True)
        axes_flat = axes.flatten() if n > 1 else [axes]

        fig.suptitle(
            f'OLMo-2-0425-1B — {domain.capitalize()} bias: Base vs. Instruct\n'
            'Abs. log prob diff per layer  (blue = States, red = Attn severed, green = MLP severed)  '
            'Y-axis fixed across all subplots.',
            fontsize=11, fontweight='bold')

        for ax, (title, s, a, m, nl, low_sig) in zip(axes_flat, subplots):
            _draw_bars(ax, s, a, m, BAR_LABELS, BAR_COLORS, nl,
                       'Layer', 'Abs. log prob diff (stereo − anti)', title)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
            if low_sig:
                ax.set_facecolor('#FFF9C4')
                ax.text(0.98, 0.97, '⚠ low-signal', transform=ax.transAxes,
                        fontsize=7, ha='right', va='top', color='#B71C1C')

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        out = os.path.join(out_dir, f'{domain}-base-vs-instruct.pdf')
        _savepdf(fig, out)


# ── main loop ─────────────────────────────────────────────────────────────────

print(f'Opening zip: {ZIP_PATH}')
zf = zipfile.ZipFile(ZIP_PATH, 'r')
zip_names_all = zf.namelist()

all_models_stats = {}  # accumulate for cross-model plots

for model_name in models_to_run:
    cfg = MODEL_CONFIGS[model_name]
    org = cfg['zip_org']

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
                loader       = _load_local
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
                loader       = lambda p: _load_zip(zf, p)
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

zf.close()

# ── base vs instruct (cross-model) ────────────────────────────────────────────
BASE     = 'OLMo-2-0425-1B'
INSTRUCT = 'OLMo-2-0425-1B-Instruct'
if RUN_COMPARE and BASE in all_models_stats and INSTRUCT in all_models_stats:
    print(f'\n  Generating base vs instruct comparison...')
    save_bias_trajectory(
        all_models_stats[BASE],
        all_models_stats[INSTRUCT],
        PLOTS_BASE,
    )
    save_base_vs_instruct(
        all_models_stats[BASE],
        all_models_stats[INSTRUCT],
        PLOTS_BASE,
    )

print('\nDone.')
