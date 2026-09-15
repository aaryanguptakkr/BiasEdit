import sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from dsets import StereoSetDataset

DATA = __file__.replace('edgecase_test.py', 'edgecase_data.json')

failures = 0
for name in ['Qwen/Qwen2.5-1.5B', 'meta-llama/Llama-3.2-1B', 'google/gemma-3-1b-pt']:
    tok = AutoTokenizer.from_pretrained(name)
    ds = StereoSetDataset(tok, DATA, name)
    print(f"\n=== {name.split('/')[-1]} ===")
    for i in range(len(ds)):
        s = ds[i]
        for kind in ('anti', 'stereo'):
            sent = s[kind]
            span = s.get(f'{kind}_blank_idxs')
            ids = tok(sent)['input_ids']
            toks = [tok.decode([t]) for t in ids]
            expect = s['attribute'][kind]
            if span is None:
                print(f"  [{s['id']}/{kind}] span=None  FAIL"); failures += 1; continue
            got = tok.decode(ids[span[0]:span[1]]).strip()
            ok = got == expect
            failures += (not ok)
            print(f"  [{s['id']:>16}/{kind:>6}] tokens={toks}")
            print(f"  {'':>27} span={span} -> {got!r} vs expected {expect!r}  {'OK' if ok else 'FAIL'}")

print(f"\n{'ALL OK' if failures == 0 else str(failures) + ' FAILURES'}")
sys.exit(1 if failures else 0)
