"""
Build a self-contained HTML analysis report with plots embedded as base64 PNGs.
Reads from stats.json files — no zip or GPU needed.

Usage:
    cd bias_tracing
    python scripts/build_html_report.py
Output:
    bias_tracing/ANALYSIS_REPORT.html
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, base64, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from plot_utils import PLOTS_BASE, BASE_MODEL, INST_MODEL, PYTHIA, BIAS_TYPES, LOW_SIGNAL

OUT_HTML = '/deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing/ANALYSIS_REPORT.html'
DOMAINS  = BIAS_TYPES

PANELS = [
    ('states_nie', 'States (full restore)'),
    ('attn_nie',   'Attn-only'),
    ('mlp_nie',    'MLP-only'),
]

# ── data loading ──────────────────────────────────────────────────────────────

def load(model):
    return json.load(open(os.path.join(PLOTS_BASE, model, 'stats.json')))['checkpoints']

base_stats     = load(BASE_MODEL)
instruct_stats = load(INST_MODEL)
pythia_stats   = load(PYTHIA)

# ── PNG helper ────────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def img_tag(b64, width='100%', caption=''):
    tag = f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:1100px;">'
    if caption:
        tag = f'<figure>{tag}<figcaption>{caption}</figcaption></figure>'
    return tag

# ── plot generators ───────────────────────────────────────────────────────────

def ckpt_colors_blues(n):
    return [cm.Blues(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]

def ckpt_colors_oranges(n):
    return [cm.Oranges(0.45 + 0.45 * i / max(n - 1, 1)) for i in range(n)]

def ckpt_colors_greens(n):
    return [cm.Greens(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]


def plot_single_bar(ckpt_stats, ckpt_label, domain):
    """Standard 3-bar chart for one checkpoint + domain."""
    e = next((e for e in ckpt_stats if e['label'] == ckpt_label), None)
    if not e:
        return None
    s = e['domains'].get(domain)
    if not s:
        return None
    nl = s['num_layers']
    xs = np.arange(nl)
    bw = 0.25
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(xs,          s['states_nie'], width=bw, color='#2196F3', edgecolor='gray', label='States (full restore)')
    ax.bar(xs + bw,     s['attn_nie'],   width=bw, color='#F44336', edgecolor='gray', label='Attn-only')
    ax.bar(xs + 2 * bw, s['mlp_nie'],    width=bw, color='#4CAF50', edgecolor='gray', label='MLP-only')
    ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
    ax.set_xlabel('Layer', fontweight='bold', fontsize=10)
    ax.set_ylabel('Abs. log prob diff (stereo − anti)', fontsize=10)
    ax.set_title(f'{domain.capitalize()} bias — {BASE_MODEL}  [{ckpt_label}]',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(xs + bw)
    ax.set_xticklabels(range(nl), fontsize=8)
    plt.tight_layout()
    return fig_to_b64(fig)


def plot_line_checkpoints(ckpt_stats, model_name, domain, colors):
    """Line plot: X=layer, one line per checkpoint."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f'{model_name} — {domain.capitalize()} bias: layer profile across checkpoints\n'
                 'Light→dark = early→late training. Each line = one checkpoint.',
                 fontsize=11, fontweight='bold')

    all_vals = []
    for key, _ in PANELS:
        for e in ckpt_stats:
            s = e['domains'].get(domain)
            if s:
                all_vals.extend(s[key])
    if not all_vals:
        plt.close(fig)
        return None
    margin = (max(all_vals) - min(all_vals)) * 0.1 or 0.01
    y_min, y_max = min(all_vals) - margin, max(all_vals) + margin

    for ax, (key, title) in zip(axes, PANELS):
        for i, e in enumerate(ckpt_stats):
            s = e['domains'].get(domain)
            if not s:
                continue
            scores = np.array(s[key])
            ls = '--' if s['effect_gap'] < LOW_SIGNAL else '-'
            ax.plot(scores, color=colors[i], linewidth=1.8, linestyle=ls, label=e['label'])
            ax.plot(0, scores[0], 'o', color=colors[i], markersize=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('Abs. log prob diff (stereo − anti)', fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
        ax.grid(alpha=0.2)

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=7, loc='upper right')
    axes[0].text(0.02, 0.03, 'dashed = low-signal (<0.03 gap)',
                 transform=axes[0].transAxes, fontsize=7, color='gray')
    return fig_to_b64(fig)


def plot_base_vs_instruct(base_stats, instruct_stats, domain):
    """Base (solid blue) vs Instruct (dashed orange) on same axes."""
    n_b = len(base_stats)
    n_i = len(instruct_stats)
    bc  = ckpt_colors_blues(n_b)
    ic  = ckpt_colors_oranges(n_i)

    all_vals = []
    for key, _ in PANELS:
        for e in base_stats + instruct_stats:
            s = e['domains'].get(domain)
            if s:
                all_vals.extend(s[key])
    if not all_vals:
        return None
    margin = (max(all_vals) - min(all_vals)) * 0.1 or 0.01
    y_min, y_max = min(all_vals) - margin, max(all_vals) + margin

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f'OLMo-2-0425-1B — {domain.capitalize()} bias: Base (solid) vs Instruct (dashed)\n'
                 'Y-axis shared. Circle = L0 for base, square = L0 for instruct.',
                 fontsize=11, fontweight='bold')

    for ax, (key, title) in zip(axes, PANELS):
        for i, e in enumerate(base_stats):
            s = e['domains'].get(domain)
            if not s:
                continue
            scores = np.array(s[key])
            ax.plot(scores, color=bc[i], linewidth=1.8, linestyle='-', label=f'[B] {e["label"]}')
            ax.plot(0, scores[0], 'o', color=bc[i], markersize=5)
        for i, e in enumerate(instruct_stats):
            s = e['domains'].get(domain)
            if not s:
                continue
            scores = np.array(s[key])
            ax.plot(scores, color=ic[i], linewidth=2.0, linestyle='--', label=f'[I] {e["label"]}')
            ax.plot(0, scores[0], 's', color=ic[i], markersize=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('Abs. log prob diff (stereo − anti)', fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)
        ax.grid(alpha=0.2)

    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=6, loc='upper right',
                    title='[B]=Base  [I]=Instruct', title_fontsize=7)
    return fig_to_b64(fig)


def plot_gap_table_fig(base_stats, instruct_stats, domain):
    """Bar chart comparing effect gap across all checkpoints: base then instruct."""
    labels, gaps, colors = [], [], []
    for e in base_stats:
        s = e['domains'].get(domain)
        labels.append(e['label'])
        gaps.append(s['effect_gap'] if s else 0)
        colors.append('#1565C0')
    for e in instruct_stats:
        s = e['domains'].get(domain)
        labels.append('[I] ' + e['label'])
        gaps.append(s['effect_gap'] if s else 0)
        colors.append('#E65100')

    fig, ax = plt.subplots(figsize=(11, 3.5))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, gaps, color=colors, edgecolor='white', width=0.6)
    ax.axvline(len(base_stats) - 0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Effect gap (high − low)', fontsize=9)
    ax.set_title(f'{domain.capitalize()} — Effect gap across all checkpoints\n'
                 'Blue = base pre-training | Orange = instruct fine-tuning | '
                 'Dashed line = phase boundary', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    return fig_to_b64(fig)


# ── generate all images ───────────────────────────────────────────────────────

print('Generating plots...')

n_base     = len(base_stats)
n_instruct = len(instruct_stats)
n_pythia   = len(pythia_stats)
bc = ckpt_colors_blues(n_base)
ic = ckpt_colors_oranges(n_instruct)
gc = ckpt_colors_greens(n_pythia)

imgs = {}

# 1. Example bar chart — explain the measurement
imgs['bar_example'] = plot_single_bar(base_stats, 's2-51B', 'gender')
print('  bar_example done')

# 2. OLMo base line plots (all 4 domains)
for d in DOMAINS:
    imgs[f'olmo_base_line_{d}'] = plot_line_checkpoints(base_stats, BASE_MODEL, d, bc)
    print(f'  olmo_base_line_{d} done')

# 3. Instruct line plots (all 4 domains)
for d in DOMAINS:
    imgs[f'olmo_inst_line_{d}'] = plot_line_checkpoints(instruct_stats, INST_MODEL, d, ic)
    print(f'  olmo_inst_line_{d} done')

# 4. Base vs instruct (all 4 domains)
for d in DOMAINS:
    imgs[f'base_vs_inst_{d}'] = plot_base_vs_instruct(base_stats, instruct_stats, d)
    print(f'  base_vs_inst_{d} done')

# 5. Effect gap bar charts (all 4 domains)
for d in DOMAINS:
    imgs[f'gap_{d}'] = plot_gap_table_fig(base_stats, instruct_stats, d)
    print(f'  gap_{d} done')

# 6. Pythia line plots (gender + race)
for d in ['gender', 'race']:
    imgs[f'pythia_line_{d}'] = plot_line_checkpoints(pythia_stats, PYTHIA, d, gc)
    print(f'  pythia_line_{d} done')


# ── HTML builder ──────────────────────────────────────────────────────────────

def section(title, anchor, content):
    return f'''
<section id="{anchor}">
  <h2>{title}</h2>
  {content}
</section>
<hr>
'''

def finding_block(number, title, content):
    return f'''
<div class="finding">
  <div class="finding-label">Finding {number}</div>
  <h3>{title}</h3>
  {content}
</div>
'''

def table(headers, rows):
    th = ''.join(f'<th>{h}</th>' for h in headers)
    trs = ''
    for row in rows:
        trs += '<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>'
    return f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'

def note(text):
    return f'<div class="note">{text}</div>'

def obs(text):
    return f'<div class="obs">{text}</div>'

CSS = '''
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Georgia, serif; font-size: 15px; line-height: 1.7;
       color: #1a1a1a; background: #fafafa; padding: 0 20px 60px; max-width: 1200px; margin: auto; }
h1   { font-size: 2em; margin: 40px 0 6px; border-bottom: 3px solid #1565C0; padding-bottom: 8px; }
h2   { font-size: 1.45em; margin: 36px 0 10px; color: #1565C0; }
h3   { font-size: 1.15em; margin: 22px 0 8px; color: #333; }
p    { margin: 10px 0; }
hr   { border: none; border-top: 1px solid #ddd; margin: 32px 0; }
ul, ol { margin: 10px 0 10px 24px; }
li   { margin: 4px 0; }
code { background: #eef; padding: 1px 5px; border-radius: 3px; font-size: 0.92em; font-family: monospace; }
pre  { background: #f4f4f4; padding: 14px; border-radius: 6px; overflow-x: auto;
       font-size: 0.88em; font-family: monospace; line-height: 1.5; }
table { border-collapse: collapse; margin: 16px 0; width: auto; font-size: 0.9em; }
th, td { border: 1px solid #ccc; padding: 7px 14px; text-align: left; }
th    { background: #e8f0fe; font-weight: bold; }
tr:nth-child(even) { background: #f8f8f8; }
figure { margin: 20px 0; }
figcaption { font-size: 0.85em; color: #555; margin-top: 6px; font-style: italic; }
.finding { background: #fff; border-left: 5px solid #1565C0;
           padding: 16px 20px; margin: 24px 0; border-radius: 0 8px 8px 0;
           box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
.finding-label { font-size: 0.78em; font-weight: bold; text-transform: uppercase;
                 letter-spacing: 1px; color: #1565C0; margin-bottom: 4px; }
.note { background: #fff8e1; border-left: 4px solid #f9a825;
        padding: 10px 16px; margin: 14px 0; border-radius: 0 6px 6px 0; font-size: 0.92em; }
.obs  { background: #e8f5e9; border-left: 4px solid #388e3c;
        padding: 10px 16px; margin: 14px 0; border-radius: 0 6px 6px 0; font-size: 0.92em; }
nav  { background: #e8f0fe; padding: 16px 22px; border-radius: 8px; margin: 24px 0;
       font-size: 0.92em; }
nav a { color: #1565C0; text-decoration: none; margin-right: 18px; }
nav a:hover { text-decoration: underline; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 16px 0; }
'''

NAV = '''
<nav>
  <strong>Contents:</strong>
  <a href="#process">1. The Process</a>
  <a href="#scores">2. Score Definitions</a>
  <a href="#graphs">3. How to Read the Graphs</a>
  <a href="#findings">4. Findings</a>
  <a href="#implications">5. Implications</a>
  <a href="#glossary">6. Glossary</a>
</nav>
'''

# ── Section 1: The Process ────────────────────────────────────────────────────

sec_process = f'''
<p>We want to know: <strong>where inside a language model is bias encoded, and does that location
change across training?</strong> The tool is <strong>causal tracing</strong> — it does not measure
whether the model is biased in general; it localises <em>which layer</em> carries the bias signal.</p>

<h3>The data — StereoSet sentence pairs</h3>
<p>Each data point is a sentence pair:</p>
<ul>
  <li><strong>Stereotyped:</strong> "The nurse said that <em>she</em> was very busy."</li>
  <li><strong>Anti-stereotyped:</strong> "The nurse said that <em>he</em> was very busy."</li>
</ul>
<p>The <strong>subject</strong> is the word driving the stereotypic prediction — here, <em>nurse</em>.
We measure whether the model assigns higher probability to the stereotyped completion.</p>

{table(
  ['Domain', 'Sentence pairs', 'Status'],
  [['Gender', '~650', 'Reliable'],
   ['Race', '~930', 'Reliable'],
   ['Profession', '~520', 'Reliable'],
   ['Religion', '~30', '⚠ Unreliable — too few pairs']]
)}

<h3>Three passes through the model</h3>
<ol>
  <li><strong>Clean run</strong> — model reads the sentence normally. Record the probability on
      the stereotyped completion token. This is <code>high_score</code>.</li>
  <li><strong>Corrupted run</strong> — add Gaussian noise to the subject token embeddings ("nurse").
      The model can no longer read who the subject is. Record the new probability. This is
      <code>low_score</code>.</li>
  <li><strong>Restoration</strong> — start from the corrupted run; at one specific
      (layer L, subject token T), replace the corrupted hidden state with the clean value from
      Pass 1. Measure how much the prediction recovers. Repeat for every layer.</li>
</ol>

<p><strong>Three restore conditions</strong> run in Pass 3:</p>
<ul>
  <li><strong>States</strong> — full restore: both MLP and Attention outputs at layer L</li>
  <li><strong>Attn-only</strong> — only Attention output restored; MLP left corrupted</li>
  <li><strong>MLP-only</strong> — only MLP output restored; Attention left corrupted</li>
</ul>
<p>This separates which computational pathway (MLP vs Attention) carries the bias at each layer.</p>
'''

# ── Section 2: Scores ─────────────────────────────────────────────────────────

sec_scores = f'''
{table(
  ['Score', 'What it is', 'Typical value (OLMo final)'],
  [
    ['<code>high_score</code>', 'Clean-run probability on the stereotyped completion. How confident the model is when it can see the subject normally.', '0.39–0.46'],
    ['<code>low_score</code>', 'Corrupted-run probability. Floor — model has lost the subject identity.', '0.30–0.34'],
    ['<code>effect_gap</code>', 'high − low. How much the model RELIES on knowing who the subject is. &lt;0.03 = unreliable signal.', '0.07–0.12'],
    ['<code>states_nie[L]</code>', 'Mean indirect effect (States restore) at layer L, averaged over all subject-token positions and all sentence pairs. This is the Y-axis of every bar and line chart.', '0.27–0.37'],
    ['<code>attn_nie[L]</code>', 'Same for Attn-only restore.', 'Low, ~0.27–0.34'],
    ['<code>mlp_nie[L]</code>', 'Same for MLP-only restore.', 'Low, ~0.27–0.34'],
  ]
)}

{note('<strong>Important:</strong> <code>states_nie</code>, <code>attn_nie</code>, <code>mlp_nie</code> are raw '
      'restoration probabilities — the same unit as the Y-axis on all charts. They are <em>not</em> '
      'normalised by the effect gap unless explicitly stated. A higher value means restoring that '
      'layer recovers more of the biased prediction.')}

<h3>What does "indirect effect declining across layers" mean?</h3>
<p>The score at layer L is highest at L0 and declines toward L15. This is <em>not</em> the
model becoming less biased at deeper layers. Bias is present throughout. The score declines
because restoring a single hidden state at a later layer creates an <strong>inconsistent internal
state</strong> — that layer's clean state is surrounded by corrupted context from all other layers
and tokens. The inconsistency limits how much recovery is possible.</p>
<p>Restoring L0 is the most upstream intervention: the correct embedding propagates forward
through all 15 subsequent layers, giving it maximum leverage. Restoring L14 is nearly too late —
the prediction is almost made, and the restored state is inconsistent with everything before it.</p>
'''

# ── Section 3: Graph guide ────────────────────────────────────────────────────

bar_ex = img_tag(imgs['bar_example'], caption=
    'Standard bar chart — OLMo final checkpoint (s2-51B), gender domain. '
    'X = layer (0–15), Y = abs. log prob diff (stereo − anti). '
    'Blue = States (full restore), Red = Effect with Attn severed, Green = Effect with MLP severed.') if imgs['bar_example'] else ''

sec_graphs = f'''
<h3>Graph type 1 — Standard bar chart (one checkpoint, one domain)</h3>
{bar_ex}
<ul>
  <li><strong>X-axis:</strong> Transformer layer. Layer 0 = the token embedding; Layer 15 = final
      transformer layer before the output head.</li>
  <li><strong>Y-axis:</strong> <code>|log_prob(stereo) − log_prob(anti)|</code> after restoring layer L.
      Higher = model more strongly prefers the stereotyped completion. Same unit as the original bar charts.</li>
  <li><strong>Blue bars (States):</strong> Full restore — the total causal effect at this layer.</li>
  <li><strong>Red bars (Attn-only):</strong> How much the Attention pathway alone contributes.</li>
  <li><strong>Green bars (MLP-only):</strong> How much the MLP pathway alone contributes.</li>
</ul>
{obs('Pattern to look for: Blue bars tallest at L0, declining toward L15. Red and green bars '
     'consistently low and flat — neither Attention nor MLP alone carries much signal.')}

<h3>Graph type 2 — Line checkpoint plot (all checkpoints, one domain)</h3>
{img_tag(imgs['olmo_base_line_gender'], caption=
    'OLMo-2-0425-1B, gender domain. Each line = one training checkpoint. '
    'Light blue = early training (0B), dark blue = late training (s2-51B). '
    'The dot on each line marks Layer 0.')}
<ul>
  <li><strong>X-axis:</strong> Layer (0–15). Same as the bar chart.</li>
  <li><strong>Y-axis:</strong> Abs. log prob diff (stereo − anti). Same unit as the bar chart — directly comparable.</li>
  <li><strong>Each line:</strong> One training checkpoint. Light = early training, dark = late training.</li>
  <li><strong>Dashed line:</strong> That checkpoint has effect gap &lt; 0.03 — estimates unreliable.</li>
  <li><strong>Three panels:</strong> States / Attn-only / MLP-only — same as the three bar colours.</li>
  <li><strong>Y-axis is shared across all three panels</strong> — you can directly compare magnitudes.</li>
</ul>
{obs('Pattern to look for: Lines shift upward as training progresses (more bias acquired) but '
     'do NOT change shape. The slope from L0 to L15 is established early and never rotates. '
     'Attn-only and MLP-only panels show flat, low lines at all checkpoints.')}

<h3>Graph type 3 — Base vs Instruct line plot</h3>
{img_tag(imgs['base_vs_inst_profession'], caption=
    'Profession domain. Blue solid = OLMo base checkpoints (light→dark = early→late). '
    'Orange dashed = Instruct fine-tuning checkpoints. Y-axis shared.')}
<ul>
  <li><strong>Blue solid lines:</strong> Base pre-training checkpoints (0B → s2-51B).</li>
  <li><strong>Orange dashed lines:</strong> Instruct fine-tuning checkpoints (step200–step2600).</li>
  <li><strong>Y-axis shared</strong> between base and instruct — positions of dashed vs solid lines
      are directly comparable.</li>
  <li><strong>Circle marker:</strong> L0 position on base lines. <strong>Square marker:</strong> L0 on instruct lines.</li>
</ul>
{obs('Pattern to look for: Instruct dashed lines sit above the final base solid line (bias '
     'strengthened). Slopes are identical (same layer structure). The gap between instruct lines '
     'is small (barely changed during instruction tuning).')}

<h3>Graph type 4 — Effect gap bar chart</h3>
{img_tag(imgs['gap_profession'], caption=
    'Effect gap (high − low) across all checkpoints for profession. '
    'Blue = base, orange = instruct. Dashed line = phase boundary.')}
<ul>
  <li><strong>X-axis:</strong> Training checkpoints in order — base first, then instruct.</li>
  <li><strong>Y-axis:</strong> Effect gap = high_score − low_score. How much the model relies on
      knowing who the subject is.</li>
  <li><strong>Dashed vertical line:</strong> Separates base pre-training from instruct fine-tuning.</li>
</ul>
{obs('Pattern to look for: Gap is near-zero at 0B, grows through pre-training, then jumps '
     'at the first instruct checkpoint and stays flat for the rest of instruction tuning.')}
'''

# ── Section 4: Findings ───────────────────────────────────────────────────────

f1_numbers = table(
  ['Domain', 'L0', 'L4', 'L8', 'L12', 'L15'],
  [['Gender',     '0.339', '0.328', '0.310', '0.301', '0.301'],
   ['Profession', '0.324', '0.311', '0.292', '0.277', '0.274'],
   ['Race',       '0.327', '0.320', '0.301', '0.293', '0.293']]
)

f2_numbers = table(
  ['Checkpoint', 'Tokens', 'L0 score', 'Effect gap'],
  [['0B',       '0',           '0.105', '0.057 (noisy)'],
   ['21B',      '21 billion',  '0.318', '0.029'],
   ['315B',     '315 billion', '0.314', '0.050'],
   ['2.4T',     '2.4 trillion','0.320', '0.068'],
   ['4T',       '4 trillion',  '0.318', '0.059'],
   ['s2-3B',    'stage 2 +3B', '0.335', '0.076'],
   ['s2-51B',   'stage 2 +51B','0.339', '0.075']]
)

f3_numbers = table(
  ['Checkpoint', 'L0', 'L8', 'L15', 'Effect gap'],
  [['Base s2-51B',      '0.324', '0.292', '0.274', '0.097'],
   ['Instruct step200',  '0.369', '0.323', '0.305', '0.119 (+23%)'],
   ['Instruct step1400', '0.371', '0.324', '0.305', '0.122 (+26%)'],
   ['Instruct step2600', '0.371', '0.324', '0.306', '0.120 (+24%)']]
)

sec_findings = f'''
{finding_block(1, 'Bias is lexically encoded at the embedding layer (L0)',
f"""
<p>For both OLMo and Pythia, at every trained checkpoint, the highest indirect effect is at
Layer 0 — the token embedding — and declines monotonically toward the final layer. Neither
MLP-only nor Attn-only scores show any peak at any layer.</p>

{f2_numbers if False else f1_numbers}
<p style="font-size:0.85em;color:#555;margin-top:4px"><em>OLMo base, final checkpoint (s2-51B), States restore.</em></p>

{img_tag(imgs['olmo_base_line_gender'], caption='OLMo base — gender: layer profile across all checkpoints. '
    'All lines peak at L0 and decline. The shape does not change across training — only the level shifts up.')}

<p><strong>Why L0?</strong> Layer 0 is the token embedding — the vector that represents the word "nurse"
or "father" before any transformer computation. This vector was learned from co-occurrence statistics
in training data: "nurse" appeared near "she" far more than near "he". The stereotypic signal is
encoded in the word's vector directly.</p>
<p>Restoring L0 gives the model back the correct word identity from the very start, allowing it to
propagate through all 15 subsequent layers. This is why it has the highest causal leverage.</p>

{note('Contrast with factual recall (ROME paper): for facts like "Eiffel Tower → Paris", the peak '
      'causal effect is at a specific mid-layer MLP. The transformer actively computes a structured '
      'lookup there. For bias, no such peak exists — the transformer propagates the stereotype without '
      'amplifying or concentrating it.')}
""")}

{finding_block(2, 'Bias is acquired early in training, then consolidated',
f"""
<p>Gender domain, OLMo base, States restore at L0:</p>
{f2_numbers}

{img_tag(imgs['gap_gender'], caption='Effect gap (how much the model relies on subject identity) '
    'across all OLMo base checkpoints, gender domain. '
    'Near-zero at 0B, jumps at 21B, grows steadily, stabilises at stage 2.')}

<p>Two distinct phases:</p>
<ol>
  <li><strong>0B → 21B:</strong> The raw L0 score jumps from 0.105 to 0.318 — a massive early leap.
      The model learned the linguistic statistics of stereotypic words very fast. But the
      <em>effect gap shrinks</em> (0.057 → 0.029) because the model does not yet strongly depend
      on the subject's identity for its biased predictions.</li>
  <li><strong>21B → 4T:</strong> The raw L0 score barely moves. The effect gap grows steadily
      (0.029 → 0.068) — the model increasingly relies on knowing who the subject is.</li>
  <li><strong>Stage 2 (s2-3B → s2-51B):</strong> Both metrics stabilise. Stage 2 training
      did not introduce new bias.</li>
</ol>

{obs('The L0-dominant declining shape is established at 21B and never changes. '
     'In the line plot, all checkpoint lines are parallel — they shift up in level but '
     'do not rotate or develop new peaks at any layer.')}
""")}

{finding_block(3, 'Instruction tuning increases bias strength but does not change its location',
f"""
<p>Profession domain (most dramatic change):</p>
{f3_numbers}

<div class="two-col">
{img_tag(imgs['base_vs_inst_profession'], width='100%',
    caption='Profession: base (solid blue) vs instruct (dashed orange). '
    'Instruct lines sit above final base line. Slopes identical.')}
{img_tag(imgs['gap_profession'], width='100%',
    caption='Effect gap across checkpoints. The jump happens at step200 and then freezes.')}
</div>

<p>Three observations:</p>
<ol>
  <li><strong>Same layer structure.</strong> The slope from L0 to L15 is identical in base and instruct.
      L0 is still the peak. Instruction tuning did not move bias to different layers.</li>
  <li><strong>Effect gap increased by ~24% for profession and race.</strong>
      The instruct model relies <em>more</em> on the subject's identity for its biased predictions.
      Instruction tuning strengthened the bias signal rather than reducing it.</li>
  <li><strong>The change happened at step200 and then froze.</strong> From step200 to step2600 the
      gap barely moves. The model recalibrated at the very start of instruction tuning and stopped
      changing.</li>
</ol>

{note('Gender shows a smaller change (~4% gap increase). Profession and race show ~24%. '
      'Religion is unreliable (too few cases).')}
""")}

{finding_block(4, 'The pattern is consistent across architectures',
f"""
<p>OLMo-2 uses a SwiGLU FFN with rotary position embeddings. Pythia uses a standard GPT-NeoX
architecture. Both show the same L0-dominant, monotonically-declining indirect effect with flat
MLP-only and Attn-only panels.</p>

<div class="two-col">
{img_tag(imgs['olmo_base_line_race'], width='100%',
    caption='OLMo base — race domain. L0 peak, declining, no mid-layer bump.')}
{img_tag(imgs['pythia_line_race'], width='100%',
    caption='Pythia — race domain. Same pattern. Steeper L0→L15 slope at late checkpoints.')}
</div>

<p><strong>Notable difference in Pythia race:</strong> At late checkpoints (step81k–143k),
the L0→L15 slope is steeper than OLMo's equivalent, and the effect gap is larger (0.112 vs 0.094).
This could reflect training data differences or architecture-specific embedding geometry — it
warrants further investigation.</p>
{obs('Core finding is architecture-independent: L0 dominance, no MLP/Attn pathway peak. '
     'The specific magnitude and slope may vary.')}
""")}

{finding_block(5, 'MLP and Attention pathways carry negligible independent bias signal',
f"""
<p>In the bar chart and line plots, the red (Attn-only) and green (MLP-only) values are
consistently low and flat across all layers, all checkpoints, and both models.</p>

{img_tag(imgs['olmo_base_line_profession'], caption=
    'OLMo base — profession domain. States panel (left) shows the L0 peak and decline. '
    'Attn-only and MLP-only panels (centre, right) show near-flat low lines — '
    'neither pathway alone carries significant bias signal at any layer.')}

<p>For factual recall, the ROME paper identifies specific MLP layers where the indirect effect
peaks sharply — those layers act as key-value stores for the fact. For bias, no such MLP peak
exists. Neither the MLP nor the Attention pathway alone has a high-indirect-effect layer.</p>
{obs('This is the mechanistic reason why ROME/MEMIT-style layer-targeted editing is unlikely '
     'to work for debiasing — there is no locatable MLP storage site to edit.')}
""")}
'''

# ── Section 5: Implications ───────────────────────────────────────────────────

sec_implications = '''
<h3>Layer-targeted editing methods (ROME, MEMIT, MEND)</h3>
<p>These methods find the causally important MLP layer via causal tracing, then modify weights
there. They work for facts because facts have a locatable MLP peak. For bias, causal tracing
finds no such peak — L0 dominance means the signal enters through the embedding, not any MLP
computation. Editing mid-layer MLPs for debiasing is targeting the wrong location.</p>

<h3>Embedding-space approaches</h3>
<p>Methods that operate on the token embedding matrix directly — Hard Debias, INLP, embedding
fine-tuning — are mechanistically better aligned with where the bias lives. The causal tracing
results provide a mechanistic justification for why embedding-level interventions are the right
target.</p>

<h3>Instruction tuning</h3>
<p>Instruction tuning (RLHF/SFT) increased the effect gap for profession and race by ~24%
rather than reducing it. It did not change the causal location of bias. Post-training alignment
does not appear to address lexically-encoded stereotypic associations and may amplify them.</p>
'''

# ── Section 6: Glossary ───────────────────────────────────────────────────────

sec_glossary = table(
  ['Term', 'Definition'],
  [
    ['Subject tokens', 'The words in the sentence whose identity drives the stereotypic prediction (e.g., "nurse", "father")'],
    ['Corruption', 'Adding Gaussian noise to subject token embeddings — breaks the model\'s ability to read who the subject is'],
    ['Restoration', 'Replacing a corrupted hidden state at one (layer, token) position with its clean value from the uncorrupted run'],
    ['Indirect effect', 'The restoration score at a given layer — how much the biased prediction recovers when only that layer is restored'],
    ['Effect gap', 'high_score − low_score — how much the model relies on the subject\'s identity. &lt;0.03 = unreliable'],
    ['L0', 'Layer 0, the embedding layer — the raw token vector before any transformer computation'],
    ['States restore', 'Full restoration at a layer — both MLP and Attention hidden states replaced with clean values'],
    ['MLP-only restore', 'Only the MLP output restored; Attention output left corrupted'],
    ['Attn-only restore', 'Only the Attention output restored; MLP output left corrupted'],
    ['Low-signal', 'Effect gap &lt; 0.03 — corruption barely changes the prediction; indirect effects are unreliable (shown as dashed lines)'],
  ]
)

# ── assemble HTML ─────────────────────────────────────────────────────────────

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Bias Tracing Analysis Report — OLMo-2-0425-1B + Pythia-1B</title>
<style>{CSS}</style>
</head>
<body>

<h1>Bias Tracing Analysis Report</h1>
<p style="color:#555;font-size:0.92em">
  OLMo-2-0425-1B (Base + Instruct) · Pythia-1B · StereoSet domains: gender, profession, race, religion
</p>

{NAV}

{section("1. The Process — What We Are Measuring", "process", sec_process)}
{section("2. Score Definitions", "scores", sec_scores)}
{section("3. How to Read the Graphs", "graphs", sec_graphs)}
{section("4. Findings", "findings", sec_findings)}
{section("5. Implications for Debiasing", "implications", sec_implications)}
{section("6. Glossary", "glossary", sec_glossary)}

</body>
</html>
'''

with open(OUT_HTML, 'w') as f:
    f.write(html)
print(f'\nSaved: {OUT_HTML}')
