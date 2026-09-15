"""How many cases does the anti/stereo token-length guard drop, and why?
Tokenizer-only replica of bias_trace.py:191 (marker glue is symmetric -> irrelevant to the delta)."""
import json, string, sys
from collections import Counter
from transformers import AutoTokenizer

DOMAINS = ["gender", "profession", "race", "religion"]
MODELS = ["allenai/OLMo-2-0425-1B", "EleutherAI/pythia-1b", "meta-llama/Llama-3.2-1B",
          "google/gemma-3-1b-pt", "gpt2"]

def fill(sentence, widx):
    w = sentence.split(" ")
    return w[widx].strip(string.punctuation) if widx < len(w) else None

for name in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:
        print(f"{name}: SKIP ({type(e).__name__})"); continue
    print(f"\n=== {name} ===")
    for dom in DOMAINS:
        try:
            data = json.load(open(f"data/domain/{dom}.json"))
        except FileNotFoundError:
            continue
        seen, tot, drop, ex = set(), 0, 0, []
        for d in data:
            if d["id"] in seen:            # race.json duplicates
                continue
            seen.add(d["id"])
            a = d["data"]["anti-stereotype"]["sentence"]
            s = d["data"]["stereotype"]["sentence"]
            la = len(tok(a)["input_ids"]); ls = len(tok(s)["input_ids"])
            tot += 1
            if la != ls:
                drop += 1
                widx = next((i for i, w in enumerate(d["context"].split(" ")) if "BLANK" in w), None)
                if widx is not None and len(ex) < 3:
                    fa, fs = fill(a, widx), fill(s, widx)
                    ex.append(f"{fa!r}({len(tok(' '+str(fa))['input_ids'])}tok) vs {fs!r}({len(tok(' '+str(fs))['input_ids'])}tok)")
        print(f"  {dom:11s} unique={tot:5d}  dropped={drop:4d} ({100*drop/max(tot,1):5.1f}%)  e.g. {' | '.join(ex)}")
