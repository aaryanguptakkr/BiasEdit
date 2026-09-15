"""Advisor's BOS-as-placeholder proposal, implemented FAIRLY — i.e. with the
legacy gpt-branch space-absorption trick (replace ' word' INCLUDING the space
with placeholder*n, n = len(encode(' word'))). Compare indices vs word_span."""
import json
import string
import sys

import os; WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
from transformers import AutoTokenizer
from dsets import StereoSetDataset

for name in ['meta-llama/Llama-3.2-1B', 'google/gemma-3-1b-pt']:
    tok = AutoTokenizer.from_pretrained(name)
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
            words = sent.split(' ')
            word = words[wi].translate(str.maketrans('', '', string.punctuation))
            ph = None
            if word and word in words[wi]:
                if wi > 0:
                    n = len(tok.encode(' ' + word, add_special_tokens=False))
                    masked = sent.replace(' ' + words[wi], ' '.join([''] ) + bos * n + words[wi].replace(word, '', 1), 1) \
                        if False else sent[:len(' '.join(words[:wi]))] + bos * n + words[wi].replace(word, '', 1) + sent[len(' '.join(words[:wi + 1])):]
                else:
                    n = len(tok.encode(word, add_special_tokens=False))
                    masked = bos * n + words[0].replace(word, '', 1) + sent[len(words[0]):]
                real_ids = tok(sent)['input_ids']
                mask_ids = tok(masked)['input_ids']
                pos = [j for j, t in enumerate(mask_ids) if t == bos_id and j > 0]
                if wi == 0 and mask_ids and mask_ids[0] == bos_id:
                    # auto-BOS at 0 plus placeholder run starting at 1
                    pos = [j for j, t in enumerate(mask_ids) if t == bos_id][1:]
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
    short = name.split('/')[-1]
    print(f'{short}: checked={checked}  identical-indices={agree}  DIFFER={disagree}  '
          f'placeholder-only-fails={ph_fail}  word_span-only-fails={ws_fail}  both-fail={both_fail}')
