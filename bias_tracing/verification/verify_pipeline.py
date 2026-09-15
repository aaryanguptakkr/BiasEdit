"""Pre-run verification of experiments/bias_trace.py + dsets/stereoset.py (current tree).

Run from bias_tracing/ with an env that has torch+transformers. CPU only, offline.
Part A: dataset + make_inputs over real tokenizers (no model forward).
Part B: numeric logic on rigged tensors.
Part C: trace_with_patch semantics on a real small model (gpt2).
"""
import os, sys, types, json, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.append("./")
sys.path.append("experiments")

import torch
import numpy as np
from transformers import AutoTokenizer

import importlib
bt = importlib.import_module("bias_trace")
from dsets import StereoSetDataset

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
def ok(msg):   print(f"{GREEN}PASS{RESET} {msg}")
def bad(msg):  print(f"{RED}FAIL{RESET} {msg}")

MODELS = {
    "OLMo-2-0425-1B": "allenai/OLMo-2-0425-1B",
    "pythia-1b":      "EleutherAI/pythia-1b",
    "Qwen2.5-1.5B":   "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B":   "meta-llama/Llama-3.2-1B",
    "gemma-3-1b-pt":  "google/gemma-3-1b-pt",
}

# ---------------------------------------------------------------- Part A
def part_a(domain_file, n_cases):
    print("\n===== Part A: dataset + make_inputs =====")
    for short, name in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            bad(f"{short}: tokenizer load failed: {e}")
            continue
        if tok.pad_token is None:               # ModelAndTokenizer does this at model-load time
            tok.pad_token = tok.eos_token
        ds = StereoSetDataset(tok, domain_file, name)
        mt = types.SimpleNamespace(tokenizer=tok, model_name=name)

        n = min(n_cases, len(ds))
        span_ok = both_spans = made = lbl_ok = subj_ok = disj_ok = 0
        import io, contextlib, re as _re
        reasons = {}
        for i in range(n):
            it = ds[i]
            for side in ("anti", "stereo"):
                bi = it[f"{side}_blank_idxs"]
                if bi is not None:
                    span_ok += 1
            if it["anti_blank_idxs"] is not None and it["stereo_blank_idxs"] is not None:
                both_spans += 1
            # exercise make_inputs exactly as main() does; capture its stdout reason
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                inp, e_range, blank_idxs, _ = bt.make_inputs(
                    mt, prompts=[it["anti"]] * 3, labels=[it["anti_mask"]] * 3,
                    subject=it["subject"], blank_idxs=it.get("anti_blank_idxs"))
            if inp is None:
                msg = buf.getvalue().strip().split("\n")[-1]
                key = _re.sub(r"['\"].*", "", msg)[:45]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            made += 1
            ids = inp["input_ids"][0]
            lbl = inp["labels"][0]
            b, e = blank_idxs
            # labels: exactly the blank span is scored
            scored = (lbl != -100).nonzero().flatten().tolist()
            if scored == list(range(b, e)):
                lbl_ok += 1
            # decode of blank span == cleaned fill word
            dec = tok.decode(ids[b:e]).strip()
            want = it["attribute"]["anti"]
            if dec == want:
                subj_ok += 1  # blank-span decodes to the fill word, in model-input coords
            else:
                reasons[f"baddecode {dec!r}!={want!r}"] = reasons.get("baddecode", 0) + 1
            # subject spans disjoint from blank span
            overlap = any(not (se <= b or ss >= e) for ss, se in e_range)
            if not overlap:
                disj_ok += 1
        print(f"-- {short} ({name})  n={n}")
        print(f"   blank span found         : {both_spans}/{n} cases (both sides)")
        print(f"   make_inputs succeeded    : {made}/{n}   (skipped {n-made})")
        print(f"   labels == blank span     : {lbl_ok}/{made}")
        print(f"   blank decode == fill word: {subj_ok}/{made}")
        print(f"   subject disjoint w/ blank: {disj_ok}/{made}")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"   skip: {k!r} x{v}")

# ---------------------------------------------------------------- Part B
def part_b():
    print("\n===== Part B: numeric logic =====")
    # B.1 causal_difference shift alignment: rig logits so position j predicts j+1
    V, T = 50, 6
    ids = torch.arange(1, T + 1).unsqueeze(0)          # tokens 1..T
    logits = torch.full((1, T, V), -10.0)
    for j in range(T - 1):
        logits[0, j, ids[0, j + 1]] = 10.0            # j confidently predicts j+1
    labels = ids.clone(); labels[:, :T - 1] = -100    # score only last real token
    # anti == stereo here -> difference must be ~0 and the score itself ~0 (perfect pred)
    d = bt.causal_difference(logits, labels, logits, labels)
    (ok if abs(d.item()) < 1e-4 else bad)(f"causal_difference aligned-pair diff ~0  (got {d.item():.2e})")
    # single scored token in the middle, aligned -> perfect pred -> score ~0
    lab1 = torch.full((1, T), -100); lab1[0, 3] = ids[0, 3]
    s = bt.causal_difference(logits, lab1, logits, lab1)
    (ok if abs(s.item()) < 1e-4 else bad)(f"causal_difference single-token aligned ~0 (got {s.item():.2e})")
    # alignment via the internal get_score: build it by calling causal_difference with a
    # zero-logit "anti" so the returned value == -(stereo mean logprob).
    zero = torch.zeros((1, T, V))
    aligned = torch.full((1, T), -100); aligned[0, 3] = ids[0, 3]          # score token 3, predicted by pos 2
    misalig = torch.full((1, T), -100); misalig[0, 3] = (ids[0, 3] + 7)   # wrong target id
    va = bt.causal_difference(zero, aligned, logits, aligned).item()      # ~ -0  (perfect)
    vm = bt.causal_difference(zero, misalig, logits, misalig).item()      # ~ -20 (confident wrong)
    (ok if abs(va) < 1e-4 and vm < -10 else bad)(
        f"get_score alignment: correct-target logprob {va:.2e}, wrong-target {vm:.2f}")

    # B.2 normalized_indirect_effect
    scores = torch.tensor([0.0, 0.5, 1.0])
    nie = bt.normalized_indirect_effect(scores, torch.tensor(1.0), torch.tensor(0.0))
    (ok if torch.allclose(nie, scores) else bad)(f"NIE identity gap=1 low=0 -> scores  (got {nie.tolist()})")
    nie2 = bt.normalized_indirect_effect(scores, torch.tensor(1.0), torch.tensor(1.0))
    (ok if torch.isnan(nie2).all() else bad)(f"NIE zero-gap -> nan  (got {nie2.tolist()})")

    # B.3 merge_token_ranges
    m = bt.merge_token_ranges([(5, 7), (1, 3), (2, 4)])
    (ok if m == [(1, 4), (5, 7)] else bad)(f"merge_token_ranges  (got {m})")

    # B.4 window layer ranges: must be symmetric & correct width for odd w
    def win(layer, w, L):
        return list(range(max(0, layer - w // 2), min(L, layer - (-w // 2))))
    cases = [
        (8, 5, 16, [6, 7, 8, 9, 10]),
        (0, 5, 16, [0, 1, 2]),
        (15, 5, 16, [13, 14, 15]),
        (13, 7, 28, [10, 11, 12, 13, 14, 15, 16]),
    ]
    for layer, w, L, want in cases:
        got = win(layer, w, L)
        (ok if got == want else bad)(f"window(layer={layer},w={w},L={L}) -> {got}  want {want}")

# ---------------------------------------------------------------- Part C
def part_c():
    print("\n===== Part C: trace_with_patch semantics (gpt2, CPU) =====")
    from transformers import AutoModelForCausalLM
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tok))
    model.eval()
    bt.nethook.set_requires_grad(False, model)

    mt = types.SimpleNamespace(model=model, tokenizer=tok, model_name=name, num_layers=model.config.n_layer)
    S = 4
    prompt = "<|endoftext|>My father is a chief operator today."
    enc = tok([prompt] * (S + 1), return_tensors="pt", padding=True)
    enc["labels"] = enc["input_ids"].clone()
    enc["labels"][:, 0] = -100
    inp = {k: v for k, v in enc.items()}

    # subject "father" span
    e_range = bt.find_token_ranges(tok, inp["input_ids"][0], "father", prompt)
    print(f"   subject span(s) for 'father': {e_range}")

    noise = 0.3
    L = model.config.n_layer

    def score(out):
        lg = bt._logits(out)
        d = bt.causal_difference(lg[1:], inp["labels"][1:], lg[1:], inp["labels"][1:])
        return d  # anti==stereo so ~0; use raw log-prob instead:
    def logprob(out):
        lg = bt._logits(out).float()
        lp = lg[:, :-1].log_softmax(-1)
        tgt = inp["labels"][:, 1:].clone()
        mask = tgt != -100
        return lp[1:][mask[1:]].gather(1, tgt[1:][mask[1:]].long().unsqueeze(1)).mean().item()

    with torch.no_grad():
        clean = logprob(model(**inp))

    # corrupted, no patch
    corr = bt.trace_with_patch(model, model, inp, states_to_patch=[],
                               tokens_to_mixs=e_range, noise=noise, replace=0)
    corr_lp = logprob(corr)

    # full restore: patch every (token, layer) residual from source clean
    layers = [bt.layername(model, l) for l in range(L)]
    src = bt.trace_source_states(model, inp, layers)
    all_states = [(t, bt.layername(model, l)) for t in range(inp["input_ids"].shape[1]) for l in range(L)]
    full = bt.trace_with_patch(model, model, inp, states_to_patch=all_states,
                               tokens_to_mixs=e_range, noise=noise, replace=0, source_cache=src)
    full_lp = logprob(full)

    print(f"   clean logprob           = {clean:.6f}")
    print(f"   corrupted (no patch)    = {corr_lp:.6f}   (delta {corr_lp-clean:+.4f})")
    print(f"   corrupted + full restore= {full_lp:.6f}   (delta {full_lp-clean:+.6f})")
    (ok if abs(full_lp - clean) < 1e-4 else bad)("full within-model restore recovers clean score")
    (ok if abs(corr_lp - clean) > 1e-3 else bad)("corruption actually moves the score")

    # source_cache vs per-call bit-exactness at one (t, L)
    st = [(3, bt.layername(model, L // 2))]
    a = bt.trace_with_patch(model, model, inp, st, e_range, noise=noise, replace=0)
    b = bt.trace_with_patch(model, model, inp, st, e_range, noise=noise, replace=0, source_cache=src)
    diff = (bt._logits(a) - bt._logits(b)).abs().max().item()
    (ok if diff == 0.0 else bad)(f"source_cache bit-exact vs per-call  (max|Δlogits|={diff:.2e})")

    # noise lands only on subject rows 1: and only subject columns
    emb_name = bt.layername(model, 0, "embed")
    seen = {}
    with torch.no_grad(), bt.nethook.TraceDict(model, [emb_name]) as td:
        model(**inp)
    base_emb = td[emb_name].output.clone()
    with torch.no_grad(), bt.nethook.TraceDict(model, [emb_name], edit_output=lambda x, layer: seen.setdefault("x", x)) as td2:
        bt.trace_with_patch(model, model, inp, [], e_range, noise=noise, replace=0)
    # can't easily capture post-edit; instead check row 0 unchanged via full output logits
    with torch.no_grad():
        base_out = model(**inp).logits
    row0_corr = bt._logits(corr)[0]
    (ok if torch.allclose(row0_corr, base_out[0], atol=1e-5) else bad)("row 0 (clean anchor) untouched by corruption")


def part_d():
    print("\n===== Part D: real ModelAndTokenizer + cross-model trace (OLMo base/instruct, CPU) =====")
    import bias_trace as _bt
    _bt.torch = torch
    src_name, tgt_name = "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-Instruct"
    mt_s = bt.ModelAndTokenizer(src_name, torch_dtype=torch.float32)
    mt_t = bt.ModelAndTokenizer(tgt_name, torch_dtype=torch.float32)
    for m in (mt_s.model, mt_t.model):
        m.to("cpu")
    print(f"   source layers={mt_s.num_layers}  target layers={mt_t.num_layers}")
    try:
        bt.validate_model_pair(mt_s, mt_t)
        ok("validate_model_pair(base, instruct) passes")
    except Exception as e:
        bad(f"validate_model_pair raised: {e}")

    ds = StereoSetDataset(mt_t.tokenizer, "data/domain/gender.json", tgt_name)
    it = next(ds[i] for i in range(len(ds))
              if ds[i]["anti_blank_idxs"] and ds[i]["stereo_blank_idxs"])
    def mk(mt, side):
        return bt.make_inputs(mt, prompts=[it[side]] * 4, labels=[it[side]] * 4,
                              subject=it["subject"], blank_idxs=it[f"{side}_blank_idxs"])
    inp_a, er_a, bi_a, _ = mk(mt_t, "anti")
    inp_s, er_s, bi_s, _ = mk(mt_t, "stereo")
    # monkeypatch cuda->cpu already handled: make_inputs sends to "cuda"; redo on cpu
    inp_a = {k: v.to("cpu") for k, v in inp_a.items()}
    inp_s = {k: v.to("cpu") for k, v in inp_s.items()}
    sl_a = bt.sentence_labels(inp_a, mt_t.tokenizer.pad_token_id)
    sl_s = bt.sentence_labels(inp_s, mt_t.tokenizer.pad_token_id)

    def sent_gap(oa, os_):
        return bt.causal_difference(bt._logits(oa)[1:], sl_a[1:], bt._logits(os_)[1:], sl_s[1:]).item()

    noise = 3.0 * bt.collect_embedding_std(mt_t, [it["subject"][0]], device="cpu")
    with torch.no_grad():
        hi = bt.causal_difference(bt._logits(mt_t.model(**inp_a)), sl_a,
                                  bt._logits(mt_t.model(**inp_s)), sl_s).item()
    lo = sent_gap(
        bt.trace_with_patch(mt_s.model, mt_t.model, inp_a, [], er_a, noise=noise),
        bt.trace_with_patch(mt_s.model, mt_t.model, inp_s, [], er_s, noise=noise))
    # within-target self-restore (source=target): patch all -> must equal hi exactly
    L = mt_t.num_layers
    layers = [bt.layername(mt_t.model, l) for l in range(L)]
    src_self_a = bt.trace_source_states(mt_t.model, inp_a, layers)
    src_self_s = bt.trace_source_states(mt_t.model, inp_s, layers)
    allst = [(t, bt.layername(mt_t.model, l))
             for t in range(inp_a["input_ids"].shape[1]) for l in range(L)]
    self_a = bt.trace_with_patch(mt_t.model, mt_t.model, inp_a, allst, er_a, noise=noise, source_cache=src_self_a)
    self_s = bt.trace_with_patch(mt_t.model, mt_t.model, inp_s, allst, er_s, noise=noise, source_cache=src_self_s)
    self_gap = sent_gap(self_a, self_s)
    # cross-model full restore (source=base): what does it land on?
    src_x_a = bt.trace_source_states(mt_s.model, inp_a, layers)
    src_x_s = bt.trace_source_states(mt_s.model, inp_s, layers)
    x_a = bt.trace_with_patch(mt_s.model, mt_t.model, inp_a, allst, er_a, noise=noise, source_cache=src_x_a)
    x_s = bt.trace_with_patch(mt_s.model, mt_t.model, inp_s, allst, er_s, noise=noise, source_cache=src_x_s)
    x_gap = sent_gap(x_a, x_s)
    with torch.no_grad():
        src_clean_gap = bt.causal_difference(bt._logits(mt_s.model(**inp_a)), sl_a,
                                             bt._logits(mt_s.model(**inp_s)), sl_s).item()

    print(f"   target clean gap (high)        = {hi:+.5f}")
    print(f"   target corrupted gap (low)     = {lo:+.5f}")
    print(f"   target self-restore-all gap    = {self_gap:+.5f}   (want == high)")
    print(f"   base-into-target restore-all   = {x_gap:+.5f}")
    print(f"   base clean gap (reference)     = {src_clean_gap:+.5f}")
    (ok if abs(self_gap - hi) < 1e-4 else bad)("within-model full restore == clean (ROME identity)")
    print("   NOTE: cross-model full restore has no a-priori target; compare it to the two clean gaps above")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parts", default="ABC")
    p.add_argument("--domain", default="data/domain/gender.json")
    p.add_argument("--n", type=int, default=150)
    a = p.parse_args()
    if "A" in a.parts: part_a(a.domain, a.n)
    if "B" in a.parts: part_b()
    if "C" in a.parts: part_c()
    if "D" in a.parts: part_d()
