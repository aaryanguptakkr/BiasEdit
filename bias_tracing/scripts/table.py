"""
Compute all numeric entries for the STORY.md comparison table.

Run from bias_tracing root:
    python scripts/table.py

Part 1 — Within-model patching (main checkpoint base vs step2000 instruct)
    K = {subject, target} token-position columns = 32 entries (16 layers × 2 positions)
    Metrics per domain (gender, profession, race):
      max(instruct_K - base_K)  — NIE and Absolute Log Prob Diff
      min(instruct_K - base_K)  — NIE and Absolute Log Prob Diff
      Pearson corr(instruct_K, base_K) — NIE and Absolute Log Prob Diff

Part 2 — Cross-patch distributional distance
    For each direction (pre_to_post, post_to_pre) and each domain:
      Compare the distribution of token-level Absolute Log Prob Diff values at
      each K position between within-model patching (target model) and
      cross-patch. Tokens are pooled across all cases (micro granularity) so the
      subject per-layer mean equals the plotted bias_mean bar in the 4-panel
      states plot.
      K = {subject, target} × 16 layers = 32 positions.
      Two metrics:
        Wasserstein-1 (W1): mean shift between distributions (units = raw delta)
        JSD: Jensen-Shannon Divergence via histogram (bits, bounded [0,1])
      Both computed per layer; reported separately for subject positions (16)
      and target positions (16), plus the combined mean over all 32 positions.

NIE  = (raw_patched_score - mean_low) / (mean_high - mean_low)
       Normalized to each model's own effect gap. Scale-invariant.
Absolute Log Prob Diff = raw patched score = |(1/N) sum_i log P(s_i | s<i; stereo)
                         - (1/N) sum_i log P(s_i | s<i; anti)|
       Mean per-token log probability difference across all N non-BOS tokens.
       Sentence-level metric (not single target-token log prob). Always >= 0.
       Not normalized by effect gap. Captures absolute magnitude changes.

K columns:
  col 0 = subject position  (bias_mean   / states_score field)
  col 2 = target  position  (blank_mean  / blank_score  field)
  col 1 = pre-target is excluded from K (context position, not knowledge-storage)

Note on Part 2 distributions:
  Distributions are over pooled subject TOKENS (micro), not per-case means, so
  their per-layer mean matches the plotted bias_mean bar.
  JSD is computed on Absolute Log Prob Diff (always >= 0), NOT on NIE (can be negative).
  Histogram uses 50 equal-width bins over the union range of both distributions.
  W1 is scale-dependent (same units as Absolute Log Prob Diff).
  JSD is scale-free (bits).
  Caveat: subject tokens within one case are correlated, not i.i.d.; this mainly
  makes the JSD histogram look slightly more confident than truly independent
  samples would. Accepted in exchange for matching the plot's aggregation.
"""

import sys
import os
import numpy as np
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zipfile
from plot_utils import (
    MODEL_CONFIGS, local_cases_dir, CROSS_PATCH_BASE, CROSS_PATCH_CONFIGS,
    collect_scores, load_npz_local, load_npz_zip, partition_names,
    zip_cases_prefix, ZIP_PATH, MAIN_ZIP,
)

DOMAINS = ['gender', 'profession', 'race']

_zf      = zipfile.ZipFile(ZIP_PATH)
_main_zf = zipfile.ZipFile(MAIN_ZIP)


def main_zip_prefix(domain):
    """Path prefix inside main.zip: main/{domain}/causal_trace/cases/"""
    return f'main/{domain}/causal_trace/cases/'


def load_K(model, org, ck_dir, domain, source):
    """
    Load subject and target position scores for one (model, checkpoint, domain).
    Returns (subj_scores, tgt_scores, mean_high, mean_low) — each of shape (16,).
    source: 'zip' | 'main_zip' | 'local'
    """
    if source == 'main_zip':
        prefix = main_zip_prefix(domain)
        names  = [n for n in _main_zf.namelist() if n.startswith(prefix) and n.endswith('.npz')]
        single, _, _ = partition_names([os.path.basename(n) for n in names])
        name_map = {os.path.basename(n): n for n in names}
        files  = [name_map[b] for b in single if b in name_map]
        loader = lambda p: load_npz_zip(_main_zf, p)
    elif source == 'zip':
        prefix = zip_cases_prefix(org, model, ck_dir, domain)
        names  = [n for n in _zf.namelist() if n.startswith(prefix) and n.endswith('.npz')]
        single, _, _ = partition_names([os.path.basename(n) for n in names])
        files  = [prefix + b for b in single]
        loader = lambda p: load_npz_zip(_zf, p)
    else:
        d = local_cases_dir(model, org, ck_dir, domain)
        fnames = [f for f in os.listdir(d) if f.endswith('.npz')]
        single, _, _ = partition_names(fnames)
        files  = [os.path.join(d, f) for f in single]
        loader = load_npz_local
    # collect_scores returns: bias_mean (subject), pre_blank_mean, blank_mean (target), n, mh, ml
    subj, _, tgt, n, mh, ml = collect_scores(files, loader)
    return subj, tgt, mh, ml


# ── Part 1: Within-model patching ────────────────────────────────────────────

print('=' * 70)
print('Part 1 — Within-model patching: main (base) vs step2000 (instruct)')
print('K = {subject, target} columns — 32 entries (16 layers × 2 positions)')
print('=' * 70)

BASE_MODEL  = 'OLMo-2-0425-1B'
BASE_CKPT   = 'main'
INST_MODEL  = 'OLMo-2-0425-1B-Instruct'
INST_CKPT   = 'step_2000'
BASE_ORG    = MODEL_CONFIGS[BASE_MODEL]['org']
INST_ORG    = MODEL_CONFIGS[INST_MODEL]['org']

results = {}

for domain in DOMAINS:
    subj_b, tgt_b, mh_b, ml_b = load_K(BASE_MODEL, BASE_ORG, BASE_CKPT, domain, 'main_zip')
    subj_i, tgt_i, mh_i, ml_i = load_K(INST_MODEL, INST_ORG, INST_CKPT, domain, 'local')

    gap_b = mh_b - ml_b
    gap_i = mh_i - ml_i

    if domain == 'gender':
        print("\n=== Part 1 RAW gender values (before NIE) — abs log prob diff, micro-avg per layer ===")
        print("Subject = bias_mean (blue 'single state' bar in the states plot).")
        print("Target  = blank_mean (green 'attribute terms' bar in the words plot).")
        print(f"mean_high (clean): base={mh_b:.6f}  instruct={mh_i:.6f}")
        print(f"mean_low  (corr ): base={ml_b:.6f}  instruct={ml_i:.6f}")
        print(f"effect_gap       : base={gap_b:.6f}  instruct={gap_i:.6f}")
        print("\nBASE (main) SUBJECT:")
        for l in range(16):
            print(f"L{l}: {subj_b[l]}")
        print("\nBASE (main) TARGET:")
        for l in range(16):
            print(f"L{l}: {tgt_b[l]}")
        print("\nINSTRUCT (step_2000) SUBJECT:")
        for l in range(16):
            print(f"L{l}: {subj_i[l]}")
        print("\nINSTRUCT (step_2000) TARGET:")
        for l in range(16):
            print(f"L{l}: {tgt_i[l]}")

    # Compact per-domain subject-position L0 summary (ALP vs NIE, all domains)
    nie_b_subj = (subj_b - ml_b) / gap_b   # (16,)
    nie_i_subj = (subj_i - ml_i) / gap_i
    print(f"\n--- {domain.upper()} subject-position summary ---")
    print(f"effect_gap        : base={gap_b:.4f}  instruct={gap_i:.4f}")
    print(f"ALP@L0 (subject)  : base={subj_b[0]:.4f}  instruct={subj_i[0]:.4f}   "
          f"[max ALP: base L{int(subj_b.argmax())}={subj_b.max():.4f}, "
          f"instruct L{int(subj_i.argmax())}={subj_i.max():.4f}]")
    print(f"NIE@L0 (subject)  : base={nie_b_subj[0]:.4f}  instruct={nie_i_subj[0]:.4f}   "
          f"[max NIE: base L{int(nie_b_subj.argmax())}={nie_b_subj.max():.4f}, "
          f"instruct L{int(nie_i_subj.argmax())}={nie_i_subj.max():.4f}]")

    # NIE: normalize each model to its own effect gap
    nie_b = np.concatenate([(subj_b - ml_b) / gap_b, (tgt_b - ml_b) / gap_b])
    nie_i = np.concatenate([(subj_i - ml_i) / gap_i, (tgt_i - ml_i) / gap_i])

    # Absolute Log Prob Diff: raw scores, no normalization
    alp_b = np.concatenate([subj_b, tgt_b])
    alp_i = np.concatenate([subj_i, tgt_i])

    diff_nie = nie_i - nie_b
    diff_alp = alp_i - alp_b

    results[domain] = {
        'max_nie': diff_nie.max(),
        'min_nie': diff_nie.min(),
        'corr_nie': pearsonr(nie_i, nie_b)[0],
        'max_alp': diff_alp.max(),
        'min_alp': diff_alp.min(),
        'corr_alp': pearsonr(alp_i, alp_b)[0],
    }

# Print table
header = f"{'Metric':<45} {'Gender':>8} {'Profession':>10} {'Race':>8}"
print(f'\n{header}')
print('-' * len(header))

metrics = [
    ('max_nie', 'max(instruct_K − base_K) — NIE'),
    ('max_alp', 'max(instruct_K − base_K) — Absolute Log Prob Diff'),
    ('min_nie', 'min(instruct_K − base_K) — NIE'),
    ('min_alp', 'min(instruct_K − base_K) — Absolute Log Prob Diff'),
    ('corr_nie', 'Pearson corr(instruct_K, base_K) — NIE'),
    ('corr_alp', 'Pearson corr(instruct_K, base_K) — Absolute Log Prob Diff'),
]

for key, label in metrics:
    vals = [results[d][key] for d in DOMAINS]
    print(f'{label:<45} {vals[0]:>8.4f} {vals[1]:>10.4f} {vals[2]:>8.4f}')

print()
print('Note: Pearson corr is identical for NIE and Absolute Log Prob Diff because')
print('NIE is a linear rescaling of Absolute Log Prob Diff (Pearson is scale-invariant).')
print()
print('Note: min(instruct_K - base_K) for Absolute Log Prob Diff > 0 for all domains.')
print('This means every single K position increased in absolute signal after post-training.')


# ── Part 2: Cross-patch distributional distance ───────────────────────────────

def load_per_token_K(files, loader):
    """
    Load token-level subject and target position Absolute Log Prob Diff scores.
    Pools EVERY subject/target token row across ALL cases (micro granularity) —
    the same aggregation collect_scores uses for the 4-panel plot bars. The
    per-layer mean of the returned subject array therefore equals the plotted
    bias_mean[l] (the blue "Effect of single state" bar) exactly. We keep the
    full pooled token rows (not per-case means) so the W1/JSD distributions are
    over the same observations the bars average.

    Subject is the position we actually report (the states plot is subject-only);
    target is still returned so callers retain the ability to compute it.

    Returns:
      subj: (T_subj, 16) — one row per subject token, pooled across all cases
      tgt:  (T_tgt,  16) — one row per target  token, pooled across all cases
      mean_high, mean_low: float scalars
    Returns (None, None, 0, 0) if no data.
    """
    subj_rows_all, tgt_rows_all, highs, lows = [], [], [], []
    for path in files:
        try:
            d = loader(path)
            scores = d['scores']  # (n_tokens, 16)
            # subject: keep every subject token row (no per-case mean)
            subj_rows = [scores[b:e] for b, e in d['corrupt_range_anti']]
            if not subj_rows:
                continue
            subj_rows_all.append(np.concatenate(subj_rows, axis=0))  # (n_subj_tok, 16)
            # target: keep every target token row
            idx0 = int(d['blank_idxs_anti'][0])
            idx1 = int(d['blank_idxs_anti'][1]) if len(d['blank_idxs_anti']) > 1 else idx0 + 1
            tgt_rows_all.append(scores[idx0:idx1])  # (n_tgt_tok, 16)
            highs.append(float(d['high_score']))
            lows.append(float(d['low_score']))
        except Exception:
            continue
    if not subj_rows_all:
        return None, None, 0.0, 0.0
    return (np.concatenate(subj_rows_all, axis=0),   # (T_subj, 16)
            np.concatenate(tgt_rows_all,  axis=0),   # (T_tgt, 16)
            float(np.mean(highs)),
            float(np.mean(lows)))


def jsd_hist(a, b, n_bins=50):
    """
    Jensen-Shannon Divergence between two 1D empirical distributions.
    Uses an equal-width histogram over the union range of both arrays, then
    scipy.spatial.distance.jensenshannon (base 2) on the bin counts.
    scipy returns the JS *distance* (sqrt of the divergence), so we square it
    to recover the JS divergence in bits, bounded [0, 1].
    Valid only when values are always >= 0 (Absolute Log Prob Diff satisfies this).
    """
    # lo = min(a.min(), b.min())
    # hi = max(a.max(), b.max())
    # if lo == hi:
        # return 0.0
    # bins = np.linspace(lo, hi, n_bins + 1)
    # pa, _ = np.histogram(a, bins=bins)
    pa, _ = np.histogram(a)
    pb, _ = np.histogram(b)
    # jensenshannon normalizes the count vectors internally and handles zero
    # bins via the 0*log0 = 0 convention; it returns the JS distance (sqrt).
    return float(jensenshannon(pa, pb, base=2) ** 2)


def k_distances(within_subj, within_tgt, cross_subj, cross_tgt):
    """
    Compute W1 and JSD for each of the 32 K positions (subject×16 + target×16).
    Returns:
      w1_subj  (16,), w1_tgt  (16,) — Wasserstein-1 per layer for each position type
      jsd_subj (16,), jsd_tgt (16,) — JSD per layer
    """
    w1_subj  = np.array([wasserstein_distance(within_subj[:, l], cross_subj[:, l]) for l in range(16)])
    w1_tgt   = np.array([wasserstein_distance(within_tgt[:,  l], cross_tgt[:,  l]) for l in range(16)])
    jsd_subj = np.array([jsd_hist(within_subj[:, l], cross_subj[:, l]) for l in range(16)])
    jsd_tgt  = np.array([jsd_hist(within_tgt[:,  l], cross_tgt[:,  l]) for l in range(16)])
    return w1_subj, w1_tgt, jsd_subj, jsd_tgt


def profile_distances(within_mean, cross_mean):
    """
    Snippet version: W1 and JSD computed directly on the per-layer mean
    profiles (the plotted bars).
        W1  = wasserstein_distance(within_mean, cross_mean)
        JSD = jensenshannon(within_mean, cross_mean, base=2) ** 2
    within_mean, cross_mean : 1-D arrays of per-layer means.
    Returns (w1, jsd) scalars.
    """
    w1  = float(wasserstein_distance(within_mean, cross_mean))
    jsd = float(jensenshannon(within_mean, cross_mean, base=2) ** 2)
    return w1, jsd


# Cross-patch directions and their within-model reference (target model)
#   pre_to_post: target = instruct (step_2000, local)
#   post_to_pre: target = base     (main,      main_zip)
DIRECTIONS = {
    'pre_to_post': {
        'label':        'Pre → Post  (base acts → instruct model)',
        'cross_dir':    CROSS_PATCH_CONFIGS['pre_to_post']['dir'],
        'within_model': INST_MODEL,
        'within_org':   INST_ORG,
        'within_ckpt':  INST_CKPT,
        'within_src':   'local',
    },
    'post_to_pre': {
        'label':        'Post → Pre  (instruct acts → base model)',
        'cross_dir':    CROSS_PATCH_CONFIGS['post_to_pre']['dir'],
        'within_model': BASE_MODEL,
        'within_org':   BASE_ORG,
        'within_ckpt':  BASE_CKPT,
        'within_src':   'main_zip',
    },
}

print()
print('=' * 70)
print('Part 2 — Cross-patch distributional distance')
print('Token scores are pooled at token-level (micro). The subject per-layer')
print('means match the 4-panel states plot bars (bias_mean) exactly.')
print('Subject and target positions are reported separately (no combined row).')
print('Metric 1: Wasserstein-1 (W1) — mean shift between distributions')
print('          units = Absolute Log Prob Diff (same scale as raw scores)')
print('Metric 2: JSD — Jensen-Shannon Divergence via histogram (bits, [0,1])')
print('=' * 70)

for dir_key, cfg in DIRECTIONS.items():
    print(f'\n── Direction: {cfg["label"]} ──')

    dir_results = {}

    for domain in DOMAINS:
        # ── Load within-model per-case scores (target model) ──────────────────
        if cfg['within_src'] == 'local':
            d_local = local_cases_dir(cfg['within_model'], cfg['within_org'],
                                      cfg['within_ckpt'], domain)
            fnames   = sorted(f for f in os.listdir(d_local) if f.endswith('.npz'))
            single, _, _ = partition_names(fnames)
            w_files  = [os.path.join(d_local, f) for f in single]
            w_loader = load_npz_local
        elif cfg['within_src'] == 'main_zip':
            prefix    = main_zip_prefix(domain)
            all_names = [n for n in _main_zf.namelist() if n.startswith(prefix) and n.endswith('.npz')]
            single, _, _ = partition_names([os.path.basename(n) for n in all_names])
            name_map  = {os.path.basename(n): n for n in all_names}
            w_files   = [name_map[b] for b in single if b in name_map]
            w_loader  = lambda p: load_npz_zip(_main_zf, p)
        else:
            prefix   = zip_cases_prefix(cfg['within_org'], cfg['within_model'],
                                        cfg['within_ckpt'], domain)
            all_names = [n for n in _zf.namelist() if n.startswith(prefix) and n.endswith('.npz')]
            single, _, _ = partition_names([os.path.basename(n) for n in all_names])
            name_map = {os.path.basename(n): n for n in all_names}
            w_files  = [name_map[b] for b in single if b in name_map]
            w_loader = lambda p: load_npz_zip(_zf, p)

        # ── Load cross-patch per-case scores ──────────────────────────────────
        c_dir   = os.path.join(CROSS_PATCH_BASE, cfg['cross_dir'],
                               domain, 'causal_trace', 'cases')
        c_fnames = sorted(f for f in os.listdir(c_dir) if f.endswith('.npz'))
        c_single, _, _ = partition_names(c_fnames)
        c_files  = [os.path.join(c_dir, f) for f in c_single]
        c_loader = load_npz_local

        w_subj, w_tgt, w_mh, w_ml = load_per_token_K(w_files, w_loader)
        c_subj, c_tgt, c_mh, c_ml = load_per_token_K(c_files, c_loader)

        if w_subj is None or c_subj is None:
            print(f'  {domain}: no data')
            continue



        if domain == 'gender':
            print(f"\n=== {dir_key} : gender SUBJECT series (token-level mean) ===")
            print("These per-layer means equal the plotted bias_mean bar exactly.")

            print("\nWithin-model (comparison target) SUBJECT:")
            for l in range(16):
                print(f"L{l}: {w_subj[:, l].mean()}")

            print("\nCross-patch SUBJECT:")
            for l in range(16):
                print(f"L{l}: {c_subj[:, l].mean()}")

            print(f"\n=== {dir_key} : gender TARGET series (token-level mean) ===")

            print("\nWithin-model (comparison target) TARGET:")
            for l in range(16):
                print(f"L{l}: {w_tgt[:, l].mean()}")

            print("\nCross-patch TARGET:")
            for l in range(16):
                print(f"L{l}: {c_tgt[:, l].mean()}")


        w1_subj, w1_tgt, jsd_subj, jsd_tgt = k_distances(w_subj, w_tgt, c_subj, c_tgt)

        # Snippet version: metrics on the per-layer mean profiles (plotted bars).
        w_subj_prof = w_subj.mean(axis=0)   # (16,) = plotted bars
        c_subj_prof = c_subj.mean(axis=0)
        w_tgt_prof  = w_tgt.mean(axis=0)
        c_tgt_prof  = c_tgt.mean(axis=0)
        snip_w1_subj, snip_jsd_subj = profile_distances(w_subj_prof, c_subj_prof)
        snip_w1_tgt,  snip_jsd_tgt  = profile_distances(w_tgt_prof,  c_tgt_prof)

        dir_results[domain] = {
            'w1_subj':  w1_subj,   # (16,)
            'w1_tgt':   w1_tgt,    # (16,)
            'jsd_subj': jsd_subj,  # (16,)
            'jsd_tgt':  jsd_tgt,   # (16,)
            'w1_subj_mean':  w1_subj.mean(),
            'w1_tgt_mean':   w1_tgt.mean(),
            'jsd_subj_mean': jsd_subj.mean(),
            'jsd_tgt_mean':  jsd_tgt.mean(),
            # Snippet version (per-layer mean profiles)
            'snip_w1_subj':  snip_w1_subj,
            'snip_jsd_subj': snip_jsd_subj,
            'snip_w1_tgt':   snip_w1_tgt,
            'snip_jsd_tgt':  snip_jsd_tgt,
            'n_within': len(w_subj),
            'n_cross':  len(c_subj),
        }

    if not dir_results:
        continue

    # ── Summary table ─────────────────────────────────────────────────────────
    hdr = f"  {'Metric':<42} {'Gender':>8} {'Profession':>12} {'Race':>8}"
    print(f'\n{hdr}')
    print('  ' + '-' * (len(hdr) - 2))

    subj_rows = [
        ('w1_subj_mean',  'W1 — subject positions (16 layers)'),
        ('jsd_subj_mean', 'JSD — subject positions (16 layers)'),
    ]
    tgt_rows = [
        ('w1_tgt_mean',   'W1 — target positions (16 layers)'),
        ('jsd_tgt_mean',  'JSD — target positions (16 layers)'),
    ]

    print('  [Subject position]')
    for key, label in subj_rows:
        vals = [dir_results.get(d, {}).get(key, float('nan')) for d in DOMAINS]
        print(f"  {label:<42} {vals[0]:>8.4f} {vals[1]:>12.4f} {vals[2]:>8.4f}")

    print('\n  [Target position]')
    for key, label in tgt_rows:
        vals = [dir_results.get(d, {}).get(key, float('nan')) for d in DOMAINS]
        print(f"  {label:<42} {vals[0]:>8.4f} {vals[1]:>12.4f} {vals[2]:>8.4f}")

    snip_rows = [
        ('snip_w1_subj',  'W1  — subject profile (16 means)'),
        ('snip_jsd_subj', 'JSD — subject profile (16 means)'),
        ('snip_w1_tgt',   'W1  — target profile (16 means)'),
        ('snip_jsd_tgt',  'JSD — target profile (16 means)'),
    ]
    print('\n  [Snippet version: metric on per-layer mean profiles (plot bars)]')
    for key, label in snip_rows:
        vals = [dir_results.get(d, {}).get(key, float('nan')) for d in DOMAINS]
        print(f"  {label:<42} {vals[0]:>8.4f} {vals[1]:>12.4f} {vals[2]:>8.4f}")

    # ── Per-layer breakdown ────────────────────────────────────────────────────
    print(f'\n  Per-layer W1 (subject position):')
    hdr2 = f"  {'Domain':<12}" + ''.join(f'  L{l:<3}' for l in range(16))
    print(hdr2)
    for domain in DOMAINS:
        if domain not in dir_results:
            continue
        vals = dir_results[domain]['w1_subj']
        row  = f"  {domain:<12}" + ''.join(f'{v:6.4f}' for v in vals)
        print(row)

    print(f'\n  Per-layer JSD (subject position):')
    print(hdr2)
    for domain in DOMAINS:
        if domain not in dir_results:
            continue
        vals = dir_results[domain]['jsd_subj']
        row  = f"  {domain:<12}" + ''.join(f'{v:6.4f}' for v in vals)
        print(row)

    print(f'\n  Per-layer W1 (target position):')
    print(hdr2)
    for domain in DOMAINS:
        if domain not in dir_results:
            continue
        vals = dir_results[domain]['w1_tgt']
        row  = f"  {domain:<12}" + ''.join(f'{v:6.4f}' for v in vals)
        print(row)

    print(f'\n  Per-layer JSD (target position):')
    print(hdr2)
    for domain in DOMAINS:
        if domain not in dir_results:
            continue
        vals = dir_results[domain]['jsd_tgt']
        row  = f"  {domain:<12}" + ''.join(f'{v:6.4f}' for v in vals)
        print(row)

print()
