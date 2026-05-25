"""
Verify the 4-panel cross-patch within-model values against table.py Part 1.

Panel (a) of the main_body 4panel  = OLMo-2-0425-1B 'main'      (main.zip)
Panel (b) of the main_body 4panel  = OLMo-2-0425-1B-Instruct 'step_2000' (local)

Both panels and table.py Part 1 read the same .npz via the same collect_scores,
so the per-layer subject (bias_mean) and target (blank_mean) means must match.
"""
import sys, os, zipfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_utils import (
    MODEL_CONFIGS, MAIN_ZIP, local_cases_dir,
    collect_scores, load_npz_local, load_npz_zip, partition_names,
)

DOMAINS = ['gender', 'profession', 'race']
_main_zf = zipfile.ZipFile(MAIN_ZIP)


def main_zip_load(domain):
    """Replicates fig.py _load_from_main_zip: OLMo-2-0425-1B 'main'."""
    prefix = f'main/{domain}/causal_trace/cases/'
    names  = [n for n in _main_zf.namelist() if n.startswith(prefix) and n.endswith('.npz')]
    basenames = [os.path.basename(n) for n in names]
    single_b, _, _ = partition_names(basenames)
    name_map = {os.path.basename(n): n for n in names}
    single   = [name_map[b] for b in single_b if b in name_map]
    loader   = lambda p: load_npz_zip(_main_zf, p)
    return collect_scores(single, loader)  # bias_mean, pre_blank, blank_mean, n, mh, ml


def local_load(model, org, ck, domain):
    """Replicates fig.py load_within_model_from_local."""
    d = local_cases_dir(model, org, ck, domain)
    fnames = sorted(f for f in os.listdir(d) if f.endswith('.npz'))
    single, _, _ = partition_names(fnames)
    files  = [os.path.join(d, f) for f in single]
    return collect_scores(files, load_npz_local)


INST_ORG = MODEL_CONFIGS['OLMo-2-0425-1B-Instruct']['org']

for domain in DOMAINS:
    print('=' * 72)
    print(f'DOMAIN: {domain}')
    print('=' * 72)

    # Panel (a): base, main.zip
    a_subj, a_pre, a_blank, a_n, a_mh, a_ml = main_zip_load(domain)
    # Panel (b): instruct, local
    b_subj, b_pre, b_blank, b_n, b_mh, b_ml = local_load(
        'OLMo-2-0425-1B-Instruct', INST_ORG, 'step_2000', domain)

    for name, subj, blank, n, mh, ml in [
        ('(a) OLMo-2-0425-1B  main      ', a_subj, a_blank, a_n, a_mh, a_ml),
        ('(b) OLMo-Instruct   step_2000 ', b_subj, b_blank, b_n, b_mh, b_ml),
    ]:
        gap = mh - ml
        print(f'\n{name}  n={n}  high={mh:.4f} low={ml:.4f} gap={gap:.4f}')
        print('  L  | subject (bias_mean)=blue States bar | target (blank_mean)=green Attr bar')
        for l in range(len(subj)):
            print(f'  L{l:<2}|   {subj[l]:+.5f}                    |   {blank[l]:+.5f}')

print('\nNote: 4panel STATES blue bar  = bias_mean (subject)  = table.py subj column')
print('      4panel WORDS green bar = blank_mean (target)  = table.py tgt  column')


# ── Cross-check: derive table.py Part 1 metrics from the SAME arrays ──────────
from scipy.stats import pearsonr
print('\n' + '=' * 72)
print('CROSS-CHECK vs table.py Part 1 (computed from the 4panel within-model arrays)')
print('=' * 72)
print(f"{'Metric':<42} {'Gender':>8} {'Profession':>11} {'Race':>8}")
res = {}
for domain in DOMAINS:
    a_subj, _, a_blank, _, a_mh, a_ml = main_zip_load(domain)
    b_subj, _, b_blank, _, b_mh, b_ml = local_load(
        'OLMo-2-0425-1B-Instruct', INST_ORG, 'step_2000', domain)
    gap_b, gap_i = a_mh - a_ml, b_mh - b_ml
    nie_b = np.concatenate([(a_subj - a_ml)/gap_b, (a_blank - a_ml)/gap_b])
    nie_i = np.concatenate([(b_subj - b_ml)/gap_i, (b_blank - b_ml)/gap_i])
    alp_b = np.concatenate([a_subj, a_blank])
    alp_i = np.concatenate([b_subj, b_blank])
    res[domain] = {
        'max_nie': (nie_i-nie_b).max(), 'min_nie': (nie_i-nie_b).min(),
        'corr_nie': pearsonr(nie_i, nie_b)[0],
        'max_alp': (alp_i-alp_b).max(), 'min_alp': (alp_i-alp_b).min(),
        'corr_alp': pearsonr(alp_i, alp_b)[0],
    }
for key, lab in [('max_nie','max(inst-base) NIE'),('max_alp','max(inst-base) ALP'),
                 ('min_nie','min(inst-base) NIE'),('min_alp','min(inst-base) ALP'),
                 ('corr_nie','corr NIE'),('corr_alp','corr ALP')]:
    v = [res[d][key] for d in DOMAINS]
    print(f'{lab:<42} {v[0]:>8.4f} {v[1]:>11.4f} {v[2]:>8.4f}')
