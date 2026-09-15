"""Counterfactual: run Gemma through the LEGACY unk-mask path (bert else-branch
semantics, which is where GemmaTokenizerFast would land) and measure survival
against the pipeline's own length-equality check + unk-span location."""
import json
import string
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('google/gemma-3-1b-pt')
unk = tok.unk_token  # '<unk>'
data = json.load(open('/deepfreeze/aag026/Aaryan2/BiasEdit/bias_tracing/data/domain/gender.json'))

total = both_ok = len_mismatch = span_wrong = 0
for obj in data:
    word_idx = next((i for i, w in enumerate(obj['context'].split(' ')) if 'BLANK' in w), None)
    if word_idx is None:
        continue
    total += 1
    ok = True
    for kind in ('anti-stereotype', 'stereotype'):
        sent = obj['data'][kind]['sentence']
        word = sent.split(' ')[word_idx].translate(str.maketrans('', '', string.punctuation))
        # bert else-branch: bare-word encode, no leading space
        n = len(tok.encode(word, add_special_tokens=False))
        masked = obj['context'].replace('BLANK', unk * n)
        real_ids = tok(sent)['input_ids']
        mask_ids = tok(masked)['input_ids']
        if len(real_ids) != len(mask_ids):
            ok = False
            len_mismatch += 1
            break
        # locate unk span in masked ids and check it aligns with the word in real ids
        unk_id = tok.unk_token_id
        pos = [i for i, t in enumerate(mask_ids) if t == unk_id]
        if not pos or tok.decode(real_ids[pos[0]:pos[-1] + 1]).strip() != word:
            ok = False
            span_wrong += 1
            break
    if ok:
        both_ok += 1

print(f"total contexts: {total}")
print(f"survive legacy unk path (len match + span aligns): {both_ok} ({100*both_ok/total:.1f}%)")
print(f"killed by length mismatch: {len_mismatch}   killed by span misalignment: {span_wrong}")
print(f"empirical word_span method (measured earlier): 1024/1026 (99.8%)")
