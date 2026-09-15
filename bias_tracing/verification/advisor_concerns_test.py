"""Verify each advisor concern against the committed production code (55e5c99).

A. word_idx via space-splitting: does context[:word_idx] == sentence[:word_idx]
   (word-for-word), i.e. does the space-split index point at the right word?
B. Capitalization: sentence-initial (word_idx==0, capitalized) span validity.
C. Overall span validity per domain per tokenizer (production StereoSetDataset).
D. Advisor's alternative — placeholder-token game using BOS (distinct from
   eos/pad) with CORRECT space-aware counting — vs our word_span: do the two
   methods agree on the indices when both succeed?
"""
import json
import string
import sys

import os; WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
from transformers import AutoTokenizer
from dsets import StereoSetDataset

DOMS = ['gender', 'profession', 'race']
MODELS = ['Qwen/Qwen2.5-1.5B', 'meta-llama/Llama-3.2-1B', 'google/gemma-3-1b-pt']

# ── A: word_idx alignment (tokenizer-independent) ─────────────────────────────
print('── A. word_idx (space-split) alignment: context prefix == sentence prefix ──')
for dom in DOMS:
    data = json.load(open(f'{WT}/data/domain/{dom}.json'))
    total = aligned = wc_match = 0
    bad_ids = []
    for obj in data:
        wi = next((i for i, w in enumerate(obj['context'].split(' ')) if 'BLANK' in w), None)
        if wi is None:
            continue
        total += 1
        ok = True
        for kind in ('anti-stereotype', 'stereotype'):
            sw = obj['data'][kind]['sentence'].split(' ')
            cw = obj['context'].split(' ')
            if len(sw) != len(cw) or sw[:wi] != cw[:wi]:
                ok = False
        aligned += ok
        wc_match += all(len(obj['data'][k]['sentence'].split(' ')) == len(obj['context'].split(' '))
                        for k in ('anti-stereotype', 'stereotype'))
        if not ok and len(bad_ids) < 3:
            bad_ids.append(obj['id'][:8])
    print(f'  {dom}: prefix-aligned {aligned}/{total} ({100*aligned/total:.1f}%)  '
          f'same-word-count {wc_match}/{total}  first misaligned ids: {bad_ids}')

# ── B + C: span validity, overall and for word_idx==0 (capitalized) ───────────
print('\n── B/C. span validity via production dataset (decode-back verified) ──')
for name in MODELS:
    tok = AutoTokenizer.from_pretrained(name)
    for dom in DOMS:
        ds = StereoSetDataset(tok, f'{WT}/data/domain/{dom}.json', name)
        tot = ok = tot0 = ok0 = 0
        for i in range(len(ds)):
            s = ds[i]
            wi = next((j for j, w in enumerate(s['context'].split(' ')) if 'BLANK' in w), None)
            valid = s.get('anti_blank_idxs') is not None and s.get('stereo_blank_idxs') is not None
            tot += 1
            ok += valid
            if wi == 0:
                tot0 += 1
                ok0 += valid
        z = f'  sentence-initial(cap): {ok0}/{tot0}' if tot0 else '  (no word_idx==0 samples)'
        print(f'  {name.split("/")[-1]:<14} {dom:<10} spans {ok}/{tot} ({100*ok/tot:.1f}%){z}')

# ── D: advisor proposal (BOS-as-placeholder, done correctly) vs word_span ─────
print('\n── D. BOS-as-placeholder (space-aware) vs word_span: index agreement ──')
name = 'meta-llama/Llama-3.2-1B'
tok = AutoTokenizer.from_pretrained(name)
assert tok.bos_token not in (tok.eos_token, tok.pad_token), 'BOS must differ from EOS/PAD'
ds = StereoSetDataset(tok, f'{WT}/data/domain/gender.json', name)
bos, bos_id = tok.bos_token, tok.bos_token_id
agree = disagree = ph_fail = ws_fail = both_fail = checked = 0
for i in range(len(ds)):
    s = ds[i]
    wi = next((j for j, w in enumerate(s['context'].split(' ')) if 'BLANK' in w), None)
    for kind in ('anti', 'stereo'):
        checked += 1
        sent = s[kind]
        ws = s.get(f'{kind}_blank_idxs')
        # placeholder method, implemented with correct space-aware counting:
        words = sent.split(' ')
        word = words[wi].translate(str.maketrans('', '', string.punctuation))
        n = len(tok.encode((' ' if wi > 0 else '') + word, add_special_tokens=False))
        masked_words = words[:wi] + [bos * n + words[wi].replace(word, '', 1) if word and word in words[wi] else bos * n] + words[wi + 1:]
        masked = ' '.join(masked_words)
        real_ids = tok(sent)['input_ids']
        mask_ids = tok(masked)['input_ids']
        pos = [j for j, t in enumerate(mask_ids) if t == bos_id and j > 0]
        ph = None
        if len(mask_ids) == len(real_ids) and pos:
            span = (pos[0], pos[-1] + 1)
            if tok.decode(real_ids[span[0]:span[1]]).strip() == word:
                ph = span
        if ph is None and ws is None:
            both_fail += 1
        elif ph is None:
            ph_fail += 1
        elif ws is None:
            ws_fail += 1
        elif ph == tuple(ws):
            agree += 1
        else:
            disagree += 1
print(f'  checked {checked} sentence-instances (gender, Llama-3.2)')
print(f'  both methods succeed AND indices identical: {agree}')
print(f'  indices DIFFER: {disagree}')
print(f'  placeholder fails where word_span succeeds: {ph_fail}')
print(f'  word_span fails where placeholder succeeds: {ws_fail}')
print(f'  both fail (skipped either way): {both_fail}')
