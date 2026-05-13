"""
Regenerate report.md for each model from the existing stats.json files.
Does not need the zip or GPU — reads from plots/{model}/stats.json only.

Usage:
    cd bias_tracing
    python scripts/regenerate_reports.py
"""

import os
import json
import datetime

PLOTS_BASE = '/deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing/plots'
BIAS_TYPES = ['gender', 'profession', 'race', 'religion']
LOW_SIGNAL_THRESHOLD = 0.03


def nie(v, low, gap):
    return (v - low) / gap if gap > 0 else 0.0


def write_report(stats, out_dir):
    model_name = stats['model']
    today      = datetime.date.today().isoformat()
    all_ckpt_stats = stats['checkpoints']

    lines = [
        f'# {model_name} — Bias Tracing Report',
        f'',
        f'Generated: {today}  ',
        f'',
        f'## What this report measures',
        f'',
        f'Causal tracing asks: *which (subject token, layer) positions causally mediate bias?*',
        f'',
        f'For each sentence pair (stereotyped vs. anti-stereotyped from StereoSet), the subject',
        f'tokens are corrupted with Gaussian noise. Then one hidden state at a time is restored to',
        f'its clean value. The **indirect effect** at (token i, layer j) = how much the prediction',
        f'recovers when only that state is restored.',
        f'',
        f'Three restore conditions per sentence pair:',
        f'- **Full restore** (single state): all components (MLP + Attn) restored at that layer',
        f'- **MLP-only**: only MLP output restored; Attn left corrupted',
        f'- **Attn-only**: only Attn output restored; MLP left corrupted',
        f'',
        f'Scores are aggregated over **subject token positions only** (not the full sentence).',
        f'',
        f'### Interpreting NIE',
        f'',
        f'NIE = (restore_score - low_score) / (high_score - low_score)',
        f'',
        f'- **NIE > 0**: this position causally helps recover the clean prediction',
        f'- **NIE < 0**: restoring this position makes things *worse* than corrupted baseline',
        f'  (inconsistent internal state from partially restoring only one position)',
        f'- **NIE = 1**: full recovery to clean-run probability',
        f'',
        f'### Key finding across OLMo-2-0425-1B and Pythia-1B',
        f'',
        f'Unlike factual recall (ROME paper), where NIE peaks sharply at specific mid-layer MLPs,',
        f'bias shows a different pattern:',
        f'',
        f'- **NIE is highest at L0 (embedding layer)** and declines monotonically — bias is',
        f'  primarily lexical, encoded in the token embedding itself (e.g. "father", "Hispanic").',
        f'- **NIE goes negative at later layers** — partial restoration creates an inconsistent',
        f'  internal state that hurts more than helps.',
        f'- **MLP-only NIE is flat or negative** — no single MLP layer acts as a bias storage site.',
        f'',
        f'Implication: intervention methods targeting specific MLP layers (ROME/MEMIT) may be',
        f'less effective for debiasing than approaches that operate on token embeddings directly.',
        f'',
        f'> **Religion domain warning**: only 24-44 sentence pairs with effect gaps often < 0.03.',
        f'> NIE estimates for religion are unreliable — treat with caution.',
        f'',
        f'---',
        f'',
        f'## Field reference',
        f'',
        f'| Field | Description |',
        f'|---|---|',
        f'| **N** | Sentence pairs processed |',
        f'| **Gap** | high_score - low_score; how much corrupting the subject hurts; < 0.03 = low-signal |',
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
            nie_l0   = nie(s['states_nie'][0],   low, gap)
            nie_lmid = nie(s['states_nie'][mid],  low, gap)
            nie_last = nie(s['states_nie'][-1],   low, gap)
            lines.append(
                f"| {e['label']} | {domain}{flag} | {s['n_cases']} | {gap:.4f} "
                f"| {s['peak_layer_states']} | {s['peak_layer_mlp']} | {s['peak_layer_attn']} "
                f"| {nie_l0:+.2f} | {nie_lmid:+.2f} | {nie_last:+.2f} |"
            )

    lines += [
        f'',
        f'---',
        f'',
        f'## NIE by layer — States (full restore)',
        f'',
        f'NIE = (restore_score - low_score) / (high_score - low_score).',
        f'Positive = recovery; negative = worse than corrupted baseline.',
        f'Rows marked ⚠ have gap < {LOW_SIGNAL_THRESHOLD} — unreliable estimates.',
        f'',
    ]

    for e in all_ckpt_stats:
        nl  = e['domains'][next(iter(e['domains']))]['num_layers'] if e['domains'] else 16
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
                flag = ' ⚠' if gap < LOW_SIGNAL_THRESHOLD else ''
                vals = ' | '.join(f'{nie(v, low, gap):+.2f}' for v in s['states_nie'])
                lines.append(f'| {domain}{flag} | {vals} |')
        lines.append('')

    lines += [
        '---',
        '',
        '## Output files',
        '',
        '```',
        f'plots/{model_name}/',
        '├── stats.json                          <- full numeric data (reload without re-running)',
        '├── report.md                           <- this file',
        '├── heatmap_checkpoint_layer.pdf        <- checkpoint x layer heatmap (MLP + Attn)',
        '├── {domain}-states-all-checkpoints.pdf <- all checkpoints in one figure (per domain)',
        '├── {domain}-words-all-checkpoints.pdf',
        '└── {label}/                            <- one folder per checkpoint',
        '    ├── {domain}-states.pdf',
        '    ├── {domain}-words.pdf',
        '    ├── composite-states.pdf',
        '    ├── composite-words.pdf',
        '    └── composite-all.pdf',
        '```',
    ]

    md_path = os.path.join(out_dir, 'report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved: {md_path}')


# ── main ──────────────────────────────────────────────────────────────────────

for model_dir in sorted(os.listdir(PLOTS_BASE)):
    stats_path = os.path.join(PLOTS_BASE, model_dir, 'stats.json')
    if not os.path.exists(stats_path):
        continue
    print(f'Processing {model_dir}...')
    stats = json.load(open(stats_path))
    write_report(stats, os.path.join(PLOTS_BASE, model_dir))

print('Done.')
