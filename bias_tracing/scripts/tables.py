"""
Paper tables for all model families, with 95% bootstrap CIs and paired tests.

Run from bias_tracing/:
    python scripts/tables.py [--families ...] [--domains ...] [--n-boot 10000] [--out tables]

Every statistic is recomputed from per-case values, resampling sentence pairs (cases).
Within a family, base, Instruct and both cross runs score the same cases, so
base-vs-Instruct and cross-vs-within comparisons are paired (same resample for both).
Delta = LPD = mean token log-prob(stereo) - mean token log-prob(anti) per case (signed);
ALP = abs. LPD = |Delta|, used only for the clean run (T4, T6).
effect gap = mean clean Delta - mean corrupted Delta (signed);
NIE = (mean restored Delta - mean corrupted Delta) / effect gap, pooled over the resampled cases.
W1/JSD (T5) compare per-case subject single-state Delta, within vs cross, averaged over layers
(--dist-chain sgn, the default). p-values are Holm-corrected across the rows of a table, except the
T5 permutation p-values, which are not corrected.

  T1 localisation        claims 1, 2, 11   (all models)
  T2 sublayer            claim 3           (all models)
  T3 post-training shape claims 4, 6       (base vs Instruct)
  T4 strength/direction  claims 5, 7, 12   (base vs Instruct)
  T5 cross-patching      claims 8, 9       (cross vs within, W1/JSD)
  T6 bias types          claim 10          (all models)
"""

import argparse
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_utils import (
    CROSS_PATCH_FAMILIES, MODEL_CONFIGS, CKPT_TICKS, PAPER_DOMAINS, LOW_SIGNAL, local_cases_dir, cross_patch_cases_dir,
    partition_names, subsample_aligned, _case_stem, load_npz_local, case_rows, result_metadata,
    disk_cache,
)

KINDS = ('state', 'pre', 'tgt', 'mlp', 'attn')
PYTHIA = ('EleutherAI/pythia-1b', 'step143000')


# ── loading ───────────────────────────────────────────────────────────────────

def first(x, default):
    return default if x is None else x


@disk_cache(lambda cases_dir, source, target: cases_dir)
def load_cases(cases_dir, source, target):
    """Per-case rows of one run: {'ids', 'abs'/'sgn': {kind: (n, L), 'high', 'low': (n,)}}."""
    single, attn, mlp = subsample_aligned(*partition_names(sorted(os.listdir(cases_dir))), None)

    def one(names):
        d, a, m = (load_npz_local(os.path.join(cases_dir, n)) for n in names)
        for x in (d, a, m):
            meta = result_metadata(x)
            if meta['source_model'] is not None and (meta['source_model'], meta['target_model']) != (source, target):
                raise ValueError(f'{cases_dir}/{names[0]}: expected {source} -> {target}, '
                                 f'found {meta["source_model"]} -> {meta["target_model"]}')
        out = {}
        for chain, signed in (('abs', False), ('sgn', True)):
            subj, pre, tgt, high, low = case_rows(d, signed)
            nan = np.full_like(tgt, np.nan)
            out[chain] = dict(state=first(subj, nan), pre=first(pre, nan), tgt=tgt,
                              mlp=first(case_rows(m, signed)[0], nan), attn=first(case_rows(a, signed)[0], nan),
                              high=high, low=low)
        return out

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, zip(single, attn, mlp)))
    print(f'    loaded {len(rows)} cases from {cases_dir}', flush=True)
    res = {'ids': [_case_stem(n) for n in single]}
    for chain in ('abs', 'sgn'):
        res[chain] = {k: np.array([r[chain][k] for r in rows], dtype=np.float64)
                      for k in KINDS + ('high', 'low')}
    return res


def within(model, ckpt, domain):
    org, name = model.split('/')
    return load_cases(local_cases_dir(name, org, ckpt, domain), model, model)


def cross(family, direction, domain):
    cfg = CROSS_PATCH_FAMILIES[family]
    src, tgt = ((cfg['base_model'], cfg['instruct_model']) if direction == 'pre_to_post'
                else (cfg['instruct_model'], cfg['base_model']))
    return load_cases(cross_patch_cases_dir(family, direction, domain), src, tgt)


def family_models(family):
    cfg = CROSS_PATCH_FAMILIES[family]
    return (cfg['base_model'], 'main'), (cfg['instruct_model'],
                                         'step_2000' if family == 'olmo_1b' else 'main')


def align(*runs):
    """Restrict runs to their shared cases, in one order."""
    common = sorted(set.intersection(*(set(r['ids']) for r in runs)))
    out = []
    for r in runs:
        pos = {c: i for i, c in enumerate(r['ids'])}
        idx = np.array([pos[c] for c in common])
        out.append({'ids': common, **{ch: {k: v[idx] for k, v in r[ch].items()}
                                      for ch in ('abs', 'sgn')}})
    return out


# ── bootstrap helpers ─────────────────────────────────────────────────────────

def weights(n, n_boot, key, seed):
    """Row 0 = the full sample (point estimate); rows 1.. = bootstrap resample counts."""
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    w = rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(np.float64)
    return np.vstack([np.ones(n), w])


def bmean(w, x):
    """Weighted mean of each column of x (cases × ...) for every weight row; NaN-aware."""
    x2 = x.reshape(len(x), -1)
    ok = ~np.isnan(x2)
    m = (w @ np.where(ok, x2, 0)) / (w @ ok)
    return m.reshape((len(w),) + x.shape[1:])


def nie(w, run, kind, scale=None):
    """Pooled signed NIE curve per weight row; scale = run whose clean/corrupted scores normalise."""
    s = (scale or run)['sgn']
    low, high = bmean(w, s['low']), bmean(w, s['high'])
    return (bmean(w, run['sgn'][kind]) - low[:, None]) / (high - low)[:, None]


def ci(v):
    """(point, lo, hi) from a weight-row vector whose row 0 is the point estimate."""
    lo, hi = np.percentile(v[1:], [2.5, 97.5])
    return float(v[0]), float(lo), float(hi)


def put(row, name, v):
    row[name], row[name + '_lo'], row[name + '_hi'] = ci(v)


def boot_p(v):
    """Two-sided bootstrap p-value that the statistic is 0."""
    b = v[1:]
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))


def holm(rows, col):
    p = np.array([r[col] for r in rows])
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    for r, a in zip(rows, adj):
        r[col + '_holm'] = float(a)


def corr_rows(a, b):
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def profile(w, run, scale=None):
    return np.concatenate([nie(w, run, k, scale) for k in KINDS], axis=1)


def jsd_hist(a, b, n_bins=50):
    """JS divergence (bits) between two samples on shared histogram edges."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=bins)
    pb, _ = np.histogram(b, bins=bins)
    return float(jensenshannon(pa, pb, base=2) ** 2)


def distance_perm(xw, xc, dist, n_perm, rng):
    """Layer-averaged distance between within and cross per-case distributions, and its
    paired-permutation null: each case's within/cross values are swapped at random."""
    def mean_dist(a, b):
        return np.mean([dist(a[:, l], b[:, l]) for l in range(a.shape[1])])
    null = np.empty(n_perm)
    for j in range(n_perm):
        swap = rng.random(len(xw)) < 0.5
        null[j] = mean_dist(np.where(swap[:, None], xc, xw), np.where(swap[:, None], xw, xc))
    return mean_dist(xw, xc), null


# ── tables ────────────────────────────────────────────────────────────────────

def t1_t2(models, domains, a):
    t1, t2 = [], []
    for label, (model, ckpt) in models:
        for d in domains:
            run = within(model, ckpt, d)
            n = len(run['ids'])
            w = weights(n, a.n_boot, f'{model}/{d}', a.seed)
            s = run['sgn']
            gap = bmean(w, s['high']) - bmean(w, s['low'])
            st, pre, tgt = (nie(w, run, k) for k in ('state', 'pre', 'tgt'))
            peak = int(np.argmax(st[0]))
            r1 = dict(model=label, domain=d, n=n, reliable=bool(abs(gap[0]) >= LOW_SIGNAL),
                      peak_layer=peak, peak_stable=float((st[1:].argmax(1) == peak).mean()))
            put(r1, 'effect_gap', gap)
            put(r1, 'NIE_subject_L0', st[:, 0])
            put(r1, 'L0_share_of_peak', st[:, 0] / st.max(1))
            put(r1, 'NIE_pretarget_last', pre[:, -1])
            put(r1, 'NIE_target_last', tgt[:, -1])
            r1['NIE_target_max_abs'] = float(np.abs(tgt[0]).max())
            t1.append(r1)
            r2 = dict(model=label, domain=d, n=n)
            for k, col in (('state', 'single_state'), ('mlp', 'MLP_window'), ('attn', 'Attn_window')):
                put(r2, f'NIE_{col}_L0', nie(w, run, k)[:, 0])
            t2.append(r2)
    return t1, t2


def t3_t4(families, domains, a):
    t3, t4 = [], []
    for fam in families:
        (bm, bc), (im, ic) = family_models(fam)
        for d in domains:
            b, i = align(within(bm, bc, d), within(im, ic, d))
            n = len(b['ids'])
            w = weights(n, a.n_boot, f'{fam}/{d}/pair', a.seed)
            pb, pi = profile(w, b), profile(w, i)
            r3 = dict(family=fam, domain=d, n=n)
            put(r3, 'r_profile', corr_rows(pb, pi))
            put(r3, 'dNIE_subject_L0', nie(w, i, 'state')[:, 0] - nie(w, b, 'state')[:, 0])
            put(r3, 'max_abs_dNIE', np.abs(pi - pb).max(1))
            t3.append(r3)

            r4 = dict(family=fam, domain=d, n=n)
            alp_b, alp_i = bmean(w[:1], b['abs']['state'])[0], bmean(w[:1], i['abs']['state'])[0]
            r4['ALP_layers_up'] = f'{int((alp_i > alp_b).sum())}/{len(alp_b)}'
            for name, xb, xi in (('ALP_single_state', b['abs']['state'].mean(1), i['abs']['state'].mean(1)),
                                 ('ALP_clean', b['abs']['high'], i['abs']['high']),
                                 ('Delta_clean', b['sgn']['high'], i['sgn']['high'])):
                diff = xi - xb
                put(r4, f'{name}_pct', 100 * (bmean(w, xi) / bmean(w, xb) - 1))
                r4[f'{name}_dz'] = float(diff.mean() / diff.std(ddof=1))
                r4[f'{name}_p_boot'] = boot_p(bmean(w, diff))
                r4[f'{name}_p_wilcoxon'] = float(wilcoxon(diff).pvalue)
            t4.append(r4)
    for name in ('ALP_single_state', 'ALP_clean', 'Delta_clean'):
        holm(t4, f'{name}_p_boot')
        holm(t4, f'{name}_p_wilcoxon')
    return t3, t4


def t5(families, domains, a):
    rows = []
    for fam in families:
        (bm, bc), (im, ic) = family_models(fam)
        for direction, tgt_model in (('pre_to_post', (im, ic)), ('post_to_pre', (bm, bc))):
            for d in domains:
                wr, cr = align(within(*tgt_model, d), cross(fam, direction, d))
                n = len(wr['ids'])
                w = weights(n, a.n_boot, f'{fam}/{direction}/{d}', a.seed)
                nw, nc = nie(w, wr, 'state'), nie(w, cr, 'state', scale=wr)
                r = dict(family=fam, direction=direction, domain=d, n=n)
                put(r, 'NIE_within_L0', nw[:, 0])
                put(r, 'NIE_cross_L0', nc[:, 0])
                put(r, 'dNIE_L0', nc[:, 0] - nw[:, 0])
                put(r, 'r_profile', corr_rows(profile(w, cr, scale=wr), profile(w, wr)))
                rng = np.random.default_rng([a.seed, zlib.crc32(f'{fam}/{direction}/{d}/dist'.encode())])
                xw, xc = wr[a.dist_chain]['state'], cr[a.dist_chain]['state']
                for name, dist in (('W1', wasserstein_distance), ('JSD', jsd_hist)):
                    obs, null = distance_perm(xw, xc, dist, a.n_perm, rng)
                    r[name], r[f'{name}_null'] = obs, float(null.mean())
                    r[f'{name}_p_perm'] = float((1 + (null >= obs).sum()) / (1 + len(null)))
                rows.append(r)
                print(f'    T5 {fam} {direction} {d} done', flush=True)
    return rows


def t6(models, domains, a):
    rows = []
    for label, (model, ckpt) in models:
        r, clean, alp = dict(model=label), {}, {}
        for d in domains:
            run = within(model, ckpt, d)
            w = weights(len(run['ids']), a.n_boot, f'{model}/{d}/types', a.seed)
            clean[d] = bmean(w, run['sgn']['high'])
            alp[d] = bmean(w, run['abs']['high'])
            put(r, f'Delta_clean_{d}', clean[d])
            put(r, f'ALP_clean_{d}', alp[d])
        if 'gender' in domains:   # gender against each other domain, on both metrics
            for name, vals in (('ALP', alp), ('Delta', clean)):
                for other in (d for d in domains if d != 'gender'):
                    diff = vals['gender'] - vals[other]
                    put(r, f'{name}_gender-{other}', diff)
                    r[f'{name}_gender-{other}_p'] = boot_p(diff)
        order = sorted(domains, key=lambda d: -clean[d][0])
        r['order'] = ' > '.join(order)
        for hi_d, lo_d in zip(order, order[1:]):
            diff = clean[hi_d] - clean[lo_d]
            put(r, f'{hi_d}-{lo_d}', diff)
            r[f'{hi_d}-{lo_d}_p'] = boot_p(diff)
        rows.append(r)
    for col in {c for r in rows for c in r if c.endswith('_p')}:
        if all(col in r for r in rows):
            holm(rows, col)
    return rows


def checkpoint_models():
    """(label, (model, checkpoint dir)) for every traced OLMo 2 and Pythia checkpoint, in training order."""
    out = []
    for name in ('OLMo-2-0425-1B', 'OLMo-2-0425-1B-Instruct', 'pythia-1b'):
        cfg = MODEL_CONFIGS[name]
        for ck_dir, label in cfg['checkpoints']:
            out.append((f"{name} {CKPT_TICKS.get(label, label).replace(chr(10), ' ')}", (f"{cfg['org']}/{name}", ck_dir)))
    return out


# ── main-body tables (read the CSVs written above) ────────────────────────────

FAMILY_NAMES = {'olmo_1b': 'OLMo 2 1B', 'qwen2.5_1.5b': 'Qwen2.5 1.5B',
                'llama3.2_1b': 'Llama 3.2 1B', 'gemma3_1b': 'Gemma 3 1B'}
MODEL_NAMES = {'OLMo-2-0425-1B': 'OLMo 2 1B Pre', 'OLMo-2-0425-1B-Instruct': 'OLMo 2 1B Post',
               'Qwen2.5-1.5B': 'Qwen2.5 1.5B Pre', 'Qwen2.5-1.5B-Instruct': 'Qwen2.5 1.5B Post',
               'Llama-3.2-1B': 'Llama 3.2 1B Pre', 'Llama-3.2-1B-Instruct': 'Llama 3.2 1B Post',
               'gemma-3-1b-pt': 'Gemma 3 1B Pre', 'gemma-3-1b-it': 'Gemma 3 1B Post',
               'pythia-1b': 'Pythia 1B'}


def main_tables(domains, out):
    """main_posttraining.tex (claim 4) from T4 and main_bias_types.tex (claim 5) from T6."""
    t4 = pd.read_csv(os.path.join(out, 'T4_posttraining_strength_direction.csv'))
    cell = lambda r, c, f: (f'{r[c]:{f}} [{r[c + "_lo"]:{f}}, {r[c + "_hi"]:{f}}]'
                            + ('$^*$' if r.get(c.replace('_pct', '') + '_p_boot_holm', 1) < 0.05 else ''))
    lines = [r'\begin{tabular}{l' + 'cc' * len(domains) + '}', r'\toprule',
             ' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{d.capitalize()}}}' for d in domains) + r' \\',
             ' '.join(rf'\cmidrule(lr){{{2 * k + 2}-{2 * k + 3}}}' for k in range(len(domains))),
             'Model & ' + ' & '.join([r'$\Delta\,\mathrm{abs.}\ LPD$ (\%)', r'$\Delta$LPD (\%)'] * len(domains)) + r' \\',
             r'\midrule']
    for fam, g in t4.groupby('family', sort=False):
        rows = {r['domain']: r for _, r in g.iterrows()}
        lines.append(FAMILY_NAMES.get(fam, fam) + ' & ' + ' & '.join(
            f'{cell(rows[d], "ALP_clean_pct", "+.1f")} & {cell(rows[d], "Delta_clean_pct", "+.1f")}'
            for d in domains) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'main_posttraining.tex'), 'w').write('\n'.join(lines) + '\n')

    t6 = pd.read_csv(os.path.join(out, 'T6_bias_types.csv'))
    order = ['gender', 'profession', 'race']
    lines = [r'\begin{tabular}{l' + 'c' * len(order) + '}', r'\toprule',
             'Model & ' + ' & '.join(d.capitalize() for d in order) + r' \\', r'\midrule']
    for _, r in t6.iterrows():
        vals = [f'{r[f"Delta_clean_{d}"]:.3f} [{r[f"Delta_clean_{d}_lo"]:.2f}, {r[f"Delta_clean_{d}_hi"]:.2f}]'
                for d in order]
        lines.append(MODEL_NAMES.get(r['model'], r['model']) + ' & ' + ' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'main_bias_types.tex'), 'w').write('\n'.join(lines) + '\n')
    print(f'    wrote main_posttraining.tex and main_bias_types.tex to {out}')


def _ci(r, c, f='.2f'):
    return f'{r[c]:{f}} [{r[c + "_lo"]:{f}}, {r[c + "_hi"]:{f}}]'


def _p(v):
    return '$<$.001' if v < 0.001 else f'{v:.3f}'


def _tabular(header, rows, path, align=None):
    align = align or 'l' * 2 + 'c' * (len(header) - 2)
    lines = [rf'\begin{{tabular}}{{{align}}}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    lines += [' & '.join(map(str, r)) + r' \\' for r in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    open(path, 'w').write('\n'.join(lines) + '\n')


def appendix_tables(out):
    """appendix_T1..T6.tex: the full tables with paper terminology, from the CSVs written above."""
    rd = lambda n: pd.read_csv(os.path.join(out, n))
    model = lambda r: MODEL_NAMES.get(r['model'], r['model'])
    fam = lambda r: FAMILY_NAMES.get(r['family'], r['family'])
    path = lambda n: os.path.join(out, n)

    t1 = rd('T1_localisation.csv').merge(rd('T2_sublayer.csv')[['model', 'domain', 'NIE_MLP_window_L0',
        'NIE_MLP_window_L0_lo', 'NIE_MLP_window_L0_hi', 'NIE_Attn_window_L0', 'NIE_Attn_window_L0_lo',
        'NIE_Attn_window_L0_hi']], on=['model', 'domain'])   # localization and sublayers in one table
    _tabular(['Model', 'Bias type', 'Effect gap', r'Single state, $\ell=0$', r'MLP window, $\ell=0$',
              r'Attn window, $\ell=0$', 'Peak layer', r'$\ell=0$ / peak', r'NIE pre-target, last $\ell$',
              r'max $|$NIE target$|$'],
             [[model(r), r['domain'].capitalize(), _ci(r, 'effect_gap', '.3f'), _ci(r, 'NIE_subject_L0'),
               _ci(r, 'NIE_MLP_window_L0'), _ci(r, 'NIE_Attn_window_L0'),
               f"{r['peak_layer']} ({100 * r['peak_stable']:.0f}\\%)", _ci(r, 'L0_share_of_peak'),
               _ci(r, 'NIE_pretarget_last'), f"{r['NIE_target_max_abs']:.2f}"] for _, r in t1.iterrows()],
             path('appendix_T1.tex'))

    t3 = rd('T3_posttraining_shape.csv')
    _tabular(['Model', 'Bias type', r'$r$ (Pre, Post)', r'$\Delta$NIE subject, $\ell=0$', r'max $|\Delta$NIE$|$'],
             [[fam(r), r['domain'].capitalize(), _ci(r, 'r_profile', '.3f'), _ci(r, 'dNIE_subject_L0'),
               _ci(r, 'max_abs_dNIE')] for _, r in t3.iterrows()], path('appendix_T3.tex'))

    t4 = rd('T4_posttraining_strength_direction.csv')
    rows = []
    for _, r in t4.iterrows():
        row = [fam(r), r['domain'].capitalize()]
        for name in ('ALP_clean', 'Delta_clean'):
            row += [_ci(r, f'{name}_pct', '+.1f'), f"{r[f'{name}_dz']:+.2f}",
                    _p(r[f'{name}_p_boot_holm']), _p(r[f'{name}_p_wilcoxon_holm'])]
        rows.append(row)
    sub = [r'$\Delta$\% [CI]', '$d_z$', '$p_\\text{boot}$', '$p_\\text{W}$']
    _tabular(['Model', 'Bias type'] + sub * 2, rows, path('appendix_T4.tex'), align='ll' + 'cccc' * 2)
    lines = open(path('appendix_T4.tex')).read().split('\n')
    lines.insert(2, r' & & \multicolumn{4}{c}{$\overline{\overline{\mathrm{abs.}\ LPD}}_\text{clean}$} & '
                    r'\multicolumn{4}{c}{$\overline{\overline{LPD}}_\text{clean}$} \\'
                    '\n\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}')
    open(path('appendix_T4.tex'), 'w').write('\n'.join(lines))

    t5 = rd('T5_cross_patching.csv')
    dirs = {'pre_to_post': r'Pre$\to$Post', 'post_to_pre': r'Post$\to$Pre'}
    _tabular(['Model', 'Direction', 'Bias type', r'NIE within, $\ell=0$', r'NIE cross, $\ell=0$', r'$\Delta$NIE, $\ell=0$',
              r'$r$ (cross, within)', 'W1 (null)', r'$p_\text{perm}$', 'JSD (null)', r'$p_\text{perm}$'],
             [[fam(r), dirs[r['direction']], r['domain'].capitalize(), _ci(r, 'NIE_within_L0'),
               _ci(r, 'NIE_cross_L0'), _ci(r, 'dNIE_L0'), _ci(r, 'r_profile'),
               f"{r['W1']:.4f} ({r['W1_null']:.4f})", _p(r['W1_p_perm']),
               f"{r['JSD']:.4f} ({r['JSD_null']:.4f})", _p(r['JSD_p_perm'])] for _, r in t5.iterrows()],
             path('appendix_T5.tex'), align='lll' + 'c' * 8)

    t6 = rd('T6_bias_types.csv')   # clean LPD per bias type and the gender differences in one table
    order = ['gender', 'profession', 'race']
    cols = [(m, o) for m in ('Delta', 'ALP') for o in ('profession', 'race')]
    _tabular(['Model'] + [f'LPD: {d}' for d in order]
             + [f'{"LPD" if m == "Delta" else "$\mathrm{abs.}\ LPD$"}: gender $-$ {o}' for m, o in cols],
             [[model(r)] + [_ci(r, f'Delta_clean_{d}', '.3f') for d in order]
              + [f"{_ci(r, f'{m}_gender-{o}', '.3f')} ({_p(r[f'{m}_gender-{o}_p_holm'])})" for m, o in cols]
              for _, r in t6.iterrows()], path('appendix_T6.tex'), align='l' + 'c' * (len(order) + len(cols)))
    if os.path.exists(path('T7_checkpoints.csv')):
        t7 = rd('T7_checkpoints.csv')
        doms = list(dict.fromkeys(t7['domain']))
        rows = []
        for ck, g in t7.groupby('model', sort=False):
            name, tick = ck.split(' ', 1)
            step, _, stage = tick.partition(' ')
            row = [f"{'Pythia' if name == 'pythia-1b' else 'OLMo 2'} {step}" + (f' ({stage})' if stage else '')]
            for d in doms:
                r = g[g['domain'] == d].iloc[0]
                ok = abs(r['effect_gap']) >= LOW_SIGNAL
                row += [f"{r['effect_gap']:.3f}"] + ([f"{r['NIE_subject_L0']:.2f}", f"{r['NIE_pretarget_last']:.2f}"]
                                                     if ok else ['--', '--'])
            rows.append(row)
        sub = ['Gap', 'Subj.', 'Pre-t.']
        _tabular(['Checkpoint'] + sub * len(doms), rows, path('appendix_T7.tex'), align='l' + 'ccc' * len(doms))
        lines = open(path('appendix_T7.tex')).read().split('\n')
        lines.insert(2, ' & ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{d.capitalize()}}}' for d in doms) + r' \\'
                        + '\n' + ' '.join(rf'\cmidrule(lr){{{3 * k + 2}-{3 * k + 4}}}' for k in range(len(doms))))
        open(path('appendix_T7.tex'), 'w').write('\n'.join(lines))
    print(f'    wrote appendix_T1..T7.tex to {out}')


# ── output ────────────────────────────────────────────────────────────────────

def emit(rows, name, out):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f'{name}.csv'), index=False)
    tex = df.copy()
    for c in [c for c in df.columns if c + '_lo' in df.columns]:
        tex[c] = [f'{v:.3f} [{lo:.3f}, {hi:.3f}]'
                  for v, lo, hi in zip(df[c], df[c + '_lo'], df[c + '_hi'])]
        tex = tex.drop(columns=[c + '_lo', c + '_hi'])
    for c in [c for c in tex.columns if c.endswith(('_p', '_holm', '_perm', '_boot', '_wilcoxon'))]:
        tex[c] = [f'{p:.1e}' if p < 1e-3 else f'{p:.3f}' for p in tex[c]]
    tex.to_latex(os.path.join(out, f'{name}.tex'), index=False, float_format='%.3f')
    print(f'\n== {name} ==\n{tex.to_string(index=False)}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--families', nargs='+', default=list(CROSS_PATCH_FAMILIES))
    ap.add_argument('--domains', nargs='+', default=PAPER_DOMAINS)
    ap.add_argument('--no-pythia', action='store_true')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--n-perm', type=int, default=1000, help='permutations for the W1/JSD test (slow)')
    ap.add_argument('--dist-chain', choices=['abs', 'sgn'], default='sgn',
                    help='per-sample LPD for the W1/JSD test: abs = |LPD|, sgn = signed LPD')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tables', nargs='+', default=['t1', 't3', 't5', 't6'],
                    help='t1 also writes t2, t3 also writes t4; t7 = checkpoints; main = LaTeX tables from the CSVs')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'tables'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    models = []
    for fam in a.families:
        (bm, bc), (im, ic) = family_models(fam)
        models += [(bm.split('/')[1], (bm, bc)), (im.split('/')[1], (im, ic))]
    if not a.no_pythia:
        models.append(('pythia-1b', PYTHIA))

    if 't1' in a.tables:
        t1, t2 = t1_t2(models, a.domains, a)
        emit(t1, 'T1_localisation', a.out)
        emit(t2, 'T2_sublayer', a.out)
    if 't3' in a.tables:
        t3, t4 = t3_t4(a.families, a.domains, a)
        emit(t3, 'T3_posttraining_shape', a.out)
        emit(t4, 'T4_posttraining_strength_direction', a.out)
    if 't5' in a.tables:
        emit(t5(a.families, a.domains, a), 'T5_cross_patching', a.out)
    if 't6' in a.tables:
        emit(t6(models, a.domains, a), 'T6_bias_types', a.out)
    if 't7' in a.tables:
        emit(t1_t2(checkpoint_models(), a.domains, a)[0], 'T7_checkpoints', a.out)
    if 'main' in a.tables:
        main_tables(a.domains, a.out)
        appendix_tables(a.out)


if __name__ == '__main__':
    main()
