"""A real case dropped by the anti/stereo length guard, token by token."""
import json, string
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", local_files_only=True)
MARK = tok.eos_token          # olmo/gpt/pythia get this glued on in make_inputs

data = json.load(open("data/domain/gender.json"))
seen = set()
for d in data:
    if d["id"] in seen: continue
    seen.add(d["id"])
    a = d["data"]["anti-stereotype"]["sentence"]; s = d["data"]["stereotype"]["sentence"]
    ia = tok(MARK + a)["input_ids"]; is_ = tok(MARK + s)["input_ids"]
    if len(ia) == len(is_):  continue
    widx = next((i for i, w in enumerate(d["context"].split(" ")) if "BLANK" in w), None)
    if widx is None: continue

    ta = [tok.decode([t]) for t in ia]; ts = [tok.decode([t]) for t in is_]
    prefix = " ".join(a.split(" ")[:widx])
    bstart = len(tok(MARK + prefix)["input_ids"])            # fill starts here in BOTH
    fa = a.split(" ")[widx].strip(string.punctuation); fs = s.split(" ")[widx].strip(string.punctuation)
    ea = len(tok(MARK + prefix + " " + fa)["input_ids"]); es = len(tok(MARK + prefix + " " + fs)["input_ids"])

    print(f"case {d['id']}   subject={d['subject']}")
    print(f"  anti  : {a}\n  stereo: {s}")
    print(f"  fill: {fa!r} = {ea-bstart} token(s)   vs   {fs!r} = {es-bstart} token(s)")
    print(f"  lengths: anti={len(ia)}  stereo={len(is_)}   -> SKIPPED by bias_trace.py\n")
    print(f"  {'idx':>3} | {'anti token':>14} | {'stereo token':>14} | same? | region")
    for i in range(max(len(ta), len(ts))):
        A = repr(ta[i]) if i < len(ta) else "—"
        S = repr(ts[i]) if i < len(ts) else "—"
        same = "yes" if (i < len(ta) and i < len(ts) and ta[i] == ts[i]) else "NO "
        if i < bstart:                     region = "shared prefix"
        elif i < max(ea, es):              region = "FILL (differs)"
        else:                              region = "suffix (shifted)"
        print(f"  {i:>3} | {A:>14} | {S:>14} |  {same}  | {region}")
    break
