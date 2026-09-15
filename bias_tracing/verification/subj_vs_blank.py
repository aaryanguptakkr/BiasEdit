"""For the cases the length-guard DROPS: where does the subject sit relative to the fill?
Cases whose subject lies entirely in the shared prefix are fully recoverable by a
prefix-aligned grid (positions < blank_start mean the same thing in both sentences)."""
import json, string
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", local_files_only=True)

for dom in ["gender", "profession", "race", "religion"]:
    data = json.load(open(f"data/domain/{dom}.json"))
    seen = set(); drop = rec = subj_after = nospan = 0
    for d in data:
        if d["id"] in seen: continue
        seen.add(d["id"])
        a = d["data"]["anti-stereotype"]["sentence"]; s = d["data"]["stereotype"]["sentence"]
        if len(tok(a)["input_ids"]) == len(tok(s)["input_ids"]): continue
        drop += 1
        widx = next((i for i, w in enumerate(d["context"].split(" ")) if "BLANK" in w), None)
        if widx is None: nospan += 1; continue
        # blank start (raw coords) and subject char position, both in the anti sentence
        prefix = " ".join(a.split(" ")[:widx])
        blank_start = len(tok(prefix)["input_ids"])
        subjs = d["subject"] if isinstance(d["subject"], list) else [d["subject"]]
        ends = [a.lower().find(str(x).lower()) + len(str(x)) for x in subjs if a.lower().find(str(x).lower()) >= 0]
        if not ends: nospan += 1; continue
        subj_tok_end = max(len(tok(a[:e])["input_ids"]) for e in ends)
        if subj_tok_end <= blank_start: rec += 1
        else: subj_after += 1
    print(f"{dom:11s} dropped={drop:4d} | subject entirely BEFORE fill (recoverable) = {rec:4d} ({100*rec/max(drop,1):.0f}%) "
          f"| subject at/after fill = {subj_after:3d} | unlocatable = {nospan:3d}")
