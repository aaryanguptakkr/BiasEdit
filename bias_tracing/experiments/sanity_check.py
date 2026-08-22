"""
Sanity check for the cross-model causal tracing pipeline (bias_trace.py).
Runs each stage independently and reports PASS/FAIL.
Writes ONLY to --output_dir (default: results/sanity_check/).
Never touches existing causal trace results.

Checks per CROSS_MODEL_CONVENTIONS.md:
  - model/tokenizer loading, layername resolution, source/target config parity
  - BOS convention (llama-3/gemma: auto-BOS; qwen/pythia: nothing; olmo/gpt: manual prepend)
  - BLANK span location (empirical length-difference for qwen/llama-3/gemma; unk-mask otherwise)
  - subject location via find_token_range, incl. multi-word subjects
  - patching machinery: full self-restore recovers the clean score; cross-model smoke test
  - end-to-end calculate_hidden_flow on --n_samples samples
  - optional --coverage: per-domain usable-sample report with skipped IDs
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # bias_tracing/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                      # experiments/

# ── helpers ────────────────────────────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}")
        return result
    except Exception:
        print(f"  {FAIL} {label}")
        traceback.print_exc()
        sys.exit(1)

# ── args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model_source",  default="allenai/OLMo-2-0425-1B")
parser.add_argument("--model_target",  default="allenai/OLMo-2-0425-1B-Instruct")
parser.add_argument("--bias_file",     default="data/domain/gender.json")
parser.add_argument("--subject_file",  default="data/knowns.json")
parser.add_argument("--output_dir",    default="results/sanity_check")
parser.add_argument("--n_samples",     default=1, type=int,
                    help="Number of dataset samples for the end-to-end trace (keep small)")
parser.add_argument("--coverage",      action="store_true",
                    help="Also report usable-sample coverage over all 4 domain files")
parser.add_argument("--domains_dir",   default="data/domain")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def family(name):
    n = name.lower()
    for k in ("qwen", "llama-3", "gemma", "olmo", "pythia", "gpt"):
        if k in n:
            return k
    return "other"

def sanity_dtype(name):
    # mirrors get_dtype() in bias_trace.main()
    import torch
    n = name.lower()
    if "olmo" in n:
        return torch.bfloat16 if "instruct" in n else torch.float32
    if "pythia" in n:
        return torch.float16
    if any(k in n for k in ("qwen", "llama-3", "gemma")):
        return torch.bfloat16
    return torch.float16

# ── stage 1: imports ───────────────────────────────────────────────────────────

print("\n[1/8] Checking imports ...")

check("numpy / torch / matplotlib",
      lambda: __import__("numpy") and __import__("torch") and __import__("matplotlib"))
check("transformers (AutoModelForCausalLM, AutoTokenizer)",
      lambda: getattr(__import__("transformers"), "AutoModelForCausalLM"))
check("dsets.StereoSetDataset",
      lambda: getattr(__import__("dsets", fromlist=["StereoSetDataset"]), "StereoSetDataset"))
check("util.nethook",
      lambda: __import__("util.nethook", fromlist=["nethook"]))

# ── stage 2: CUDA ─────────────────────────────────────────────────────────────

print("\n[2/8] Checking CUDA ...")

import torch

def _cuda_check():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"         device count={torch.cuda.device_count()}  device[0]={torch.cuda.get_device_name(0)}")
    return True

check("CUDA available + device info", _cuda_check)

# ── stage 3: models + config parity ───────────────────────────────────────────

print(f"\n[3/8] Loading models ...\n       source: {args.model_source}\n       target: {args.model_target}")

from bias_trace import ModelAndTokenizer, layername, make_inputs, collect_embedding_std, \
    calculate_hidden_flow, trace_with_patch, find_complete_word, find_token_range, \
    decode_tokens, _logits, causal_difference

def _load(name):
    mt = ModelAndTokenizer(name, torch_dtype=sanity_dtype(name))
    assert mt.iscausal, f"{name} must be treated as causal LM"
    assert mt.num_layers > 0, "num_layers must be > 0"
    l0, le = layername(mt.model, 0), layername(mt.model, 0, "embed")
    la, lm = layername(mt.model, 0, "attn"), layername(mt.model, 0, "mlp")
    mods = dict(mt.model.named_modules())
    for l in (l0, le, la, lm):
        assert l in mods, f"layername {l} not found in model modules"
    print(f"         {name}: iscausal={mt.iscausal} layers={mt.num_layers} "
          f"dtype={sanity_dtype(name)}  layer0={l0} attn={la} mlp={lm}")
    return mt

mt_s = check(f"source loads ({args.model_source})", lambda: _load(args.model_source))
mt_t = check(f"target loads ({args.model_target})", lambda: _load(args.model_target))

def _parity():
    cs, ct = mt_s.model.config, mt_t.model.config
    assert cs.hidden_size == ct.hidden_size, f"hidden_size {cs.hidden_size} != {ct.hidden_size}"
    assert mt_s.num_layers == mt_t.num_layers, f"num_layers {mt_s.num_layers} != {mt_t.num_layers}"
    probe = "The doctor said she was busy."
    assert mt_s.tokenizer(probe)["input_ids"] == mt_t.tokenizer(probe)["input_ids"], \
        "source/target tokenizers disagree — cross-patching invalid"
    print(f"         hidden={cs.hidden_size} layers={mt_s.num_layers} tokenizers identical")
    return True

check("source/target config + tokenizer parity", _parity)

# ── stage 4: BOS convention ───────────────────────────────────────────────────

print("\n[4/8] Checking BOS convention ...")

def _bos_check():
    tok = mt_t.tokenizer
    fam = family(args.model_target)
    probe = "The doctor is here"
    with_sp = tok(probe)["input_ids"]
    no_sp = tok(probe, add_special_tokens=False)["input_ids"]
    if fam in ("llama-3", "gemma"):
        assert len(with_sp) == len(no_sp) + 1 and with_sp[0] == tok.bos_token_id, \
            f"{fam}: expected auto-BOS at position 0 (got {with_sp[:3]} vs {no_sp[:3]})"
        print(f"         {fam}: tokenizer auto-adds BOS id={with_sp[0]} ✓ (per TransformerLens/vendor convention)")
    elif fam in ("qwen", "pythia"):
        assert with_sp == no_sp, f"{fam}: expected NO token prepended (got {with_sp[:3]} vs {no_sp[:3]})"
        print(f"         {fam}: nothing prepended ✓ (per Qwen team / TransformerLens default_prepend_bos=False)")
    else:
        print(f"         {fam}: manual prepend inside make_inputs (existing convention, unchecked here)")
    return True

check("position-0 convention matches CROSS_MODEL_CONVENTIONS.md", _bos_check)

# ── stage 5: dataset + BLANK spans + subject location ─────────────────────────

print(f"\n[5/8] Dataset, BLANK spans, subject location ({args.bias_file}) ...")

from dsets import StereoSetDataset

ds = check("StereoSetDataset loads", lambda: StereoSetDataset(mt_t.tokenizer, args.bias_file, args.model_target))

def _blank_spans():
    tok = mt_t.tokenizer
    n_check = min(20, len(ds))
    ok, bad, shown = 0, 0, 0
    for i in range(n_check):
        s = ds[i]
        if ds.use_empirical_blank:
            spans = (s.get("anti_blank_idxs"), s.get("stereo_blank_idxs"))
            sents = (s["anti"], s["stereo"])
            words = (s["attribute"]["anti"], s["attribute"]["stereo"])
        else:
            # legacy unk-mask path: verify the masked sentence still contains the mask token
            spans, sents, words = None, None, None
            ok += int(ds.mask_token in s["anti_mask"] and ds.mask_token in s["stereo_mask"])
            continue
        for span, sent, word in zip(spans, sents, words):
            if span is None:
                bad += 1
                continue
            got = tok.decode(tok(sent)["input_ids"][span[0]:span[1]]).strip()
            assert got == word, f"span decode mismatch: {got!r} != {word!r} in {sent!r}"
            ok += 1
            if shown < 3:
                print(f"         id={s['id']} span={span} decodes to {got!r} ✓")
                shown += 1
    assert ok > 0, f"no valid BLANK spans in first {n_check} samples (bad={bad})"
    print(f"         spans valid: {ok}  unresolvable(skip-on-run): {bad}  (first {n_check} samples)")
    return True

check("BLANK span location (decode-back verified)", _blank_spans)

def _subject_location():
    tok = mt_t.tokenizer
    n_check, found, missed, absent = min(20, len(ds)), 0, 0, 0
    for i in range(n_check):
        s = ds[i]
        ids = torch.tensor(tok(s["anti"])["input_ids"])
        for subj in s["subject"]:
            if find_complete_word(s["anti"], subj) is None:
                absent += 1
                continue
            r = find_token_range(tok, ids, subj, s["anti"])
            found += int(r is not None)
            missed += int(r is None)
    # explicit multi-word test (flagged SentencePiece risk for gemma)
    mw_sent, mw_subj = "People from the Middle East are kind", "Middle East"
    mw_ids = torch.tensor(tok(mw_sent)["input_ids"])
    mw = find_token_range(tok, mw_ids, mw_subj, mw_sent)
    assert mw is not None, f"find_token_range failed on multi-word subject {mw_subj!r}"
    got = tok.decode(mw_ids[mw[0]:mw[1]]).strip()
    assert mw_subj in got, f"multi-word subject span decodes to {got!r}"
    assert found > 0 and missed == 0, f"subject location: found={found} missed={missed}"
    print(f"         subjects located: {found}/{found + missed}; absent={absent}; "
          f"multi-word {mw_subj!r} -> {mw} ({got!r}) ✓")
    return True

check("subject location via find_token_range (incl. multi-word)", _subject_location)

# ── stage 6: make_inputs + noise ──────────────────────────────────────────────

print("\n[6/8] make_inputs + noise level ...")

def _first_usable(dataset):
    for s in dataset:
        ia = make_inputs(mt_t, prompts=[s["anti"]] * 2, labels=[s["anti_mask"]] * 2,
                         subject=s["subject"], blank_idxs=s.get("anti_blank_idxs"))
        is_ = make_inputs(mt_t, prompts=[s["stereo"]] * 2, labels=[s["stereo_mask"]] * 2,
                          subject=s["subject"], blank_idxs=s.get("stereo_blank_idxs"))
        if ia[0] is None or is_[0] is None:
            continue
        if ia[0]["input_ids"].shape[1] != is_[0]["input_ids"].shape[1]:
            continue
        return s, ia, is_
    raise AssertionError("no usable sample found")

sample, (inp_a, e_a, bi_a, inp_a_o), (inp_s, e_s, bi_s, inp_s_o) = \
    check("make_inputs on first usable sample", lambda: _first_usable(ds))

def _inputs_detail():
    toks = decode_tokens(mt_t.tokenizer, inp_a["input_ids"][0])
    print(f"         anti tokens: {toks}")
    print(f"         subject e_range={e_a}  blank_idxs={bi_a}")
    for (b, e) in e_a:
        assert 0 <= b < e <= len(toks), f"subject range {b, e} out of bounds"
    b, e = bi_a
    assert 0 <= b < e <= len(toks), f"blank range {b, e} out of bounds"
    scored = torch.where(inp_a["labels"][0] != -100)[0].tolist()
    assert scored == list(range(b, e)), f"labels score {scored}, expected BLANK span {b, e}"
    return True

check("ranges in bounds", _inputs_detail)

noise_level = check("collect_embedding_std on target (20 subjects)", lambda: (
    lambda nl: (print(f"         noise_level={nl:.6f}"), nl)[1]
)(3.0 * collect_embedding_std(mt_t, json.load(open(args.subject_file))[:20])))

# ── stage 7: patching machinery ───────────────────────────────────────────────

print("\n[7/8] Patching machinery ...")

def _scores(pred_a, pred_s):
    return causal_difference(pred_a, inp_a["labels"], pred_s, inp_s["labels"])

def _self_restore():
    ntoks = inp_a["input_ids"].shape[1]
    all_states = [(t, layername(mt_t.model, L)) for t in range(ntoks) for L in range(mt_t.num_layers)]
    with torch.no_grad():
        base = _scores(_logits(mt_t.model(**inp_a)), _logits(mt_t.model(**inp_s)))
    low_a = trace_with_patch(mt_t.model, mt_t.model, inp_a, [], e_a, noise=noise_level)
    low_s = trace_with_patch(mt_t.model, mt_t.model, inp_s, [], e_s, noise=noise_level)
    low = causal_difference(_logits(low_a)[1:], inp_a["labels"][1:], _logits(low_s)[1:], inp_s["labels"][1:])
    full_a = trace_with_patch(mt_t.model, mt_t.model, inp_a, all_states, e_a, noise=noise_level)
    full_s = trace_with_patch(mt_t.model, mt_t.model, inp_s, all_states, e_s, noise=noise_level)
    full = causal_difference(_logits(full_a)[1:], inp_a["labels"][1:], _logits(full_s)[1:], inp_s["labels"][1:])
    gap, recovered = abs(base.item() - low.item()), abs(base.item() - full.item())
    print(f"         base={base.item():.4f}  corrupted={low.item():.4f}  full-restore={full.item():.4f}")
    if gap < 1e-4:
        print(f"  {WARN} corruption barely moved the score (gap={gap:.6f}) — noise may be ineffective on this sample")
    else:
        assert recovered < 0.1 * gap + 1e-3, \
            f"full self-restore did not recover clean score: |base-full|={recovered:.4f} vs gap={gap:.4f}"
    return True

check("full self-restore recovers clean score (hooks verified end-to-end)", _self_restore)

def _cross_smoke():
    out = trace_with_patch(mt_s.model, mt_t.model, inp_a,
                           [(1, layername(mt_t.model, 0))], e_a, noise=noise_level)
    lg = _logits(out)
    assert torch.isfinite(lg).all(), "non-finite logits in cross-model patch"
    print(f"         source->target single-state patch: logits {tuple(lg.shape)} finite ✓")
    return True

check("cross-model patch smoke test (source -> target)", _cross_smoke)

# ── stage 8: end-to-end trace ─────────────────────────────────────────────────

print(f"\n[8/8] End-to-end causal trace ({args.n_samples} sample(s)) ...")

import numpy

traced, skipped = 0, 0
for knowledge in ds:
    if traced >= args.n_samples:
        break
    ia = make_inputs(mt_t, prompts=[knowledge["anti"]] * 2, labels=[knowledge["anti_mask"]] * 2,
                     subject=knowledge["subject"], blank_idxs=knowledge.get("anti_blank_idxs"))
    is_ = make_inputs(mt_t, prompts=[knowledge["stereo"]] * 2, labels=[knowledge["stereo_mask"]] * 2,
                      subject=knowledge["subject"], blank_idxs=knowledge.get("stereo_blank_idxs"))
    if ia[0] is None or is_[0] is None or ia[0]["input_ids"].shape[1] != is_[0]["input_ids"].shape[1]:
        skipped += 1
        continue
    result = calculate_hidden_flow(
        mt_s, mt_t, knowledge,
        ia[0], is_[0], ia[1], is_[1], ia[2], is_[2], ia[3], is_[3],
        noise=noise_level, kind=None,
    )
    if not result:
        skipped += 1
        continue
    out_path = os.path.join(args.output_dir, f"sanity_{knowledge['id']}.npz")
    numpy.savez(out_path, **{
        k: v.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(v) else v
        for k, v in result.items()
    })
    traced += 1
    print(f"  {PASS} traced {traced}/{args.n_samples}: {knowledge['id']} -> {out_path}")

if traced == 0:
    print(f"  {FAIL} no samples traced (skipped={skipped})")
    sys.exit(1)

# ── optional: per-domain coverage ─────────────────────────────────────────────

if args.coverage:
    print("\n[coverage] usable-sample report per domain ...")
    tok = mt_t.tokenizer
    report = {}
    for dom in sorted(os.listdir(args.domains_dir)):
        if not dom.endswith(".json"):
            continue
        dds = StereoSetDataset(tok, os.path.join(args.domains_dir, dom), args.model_target)
        usable, skipped_ids = 0, []
        for i in range(len(dds)):
            s = dds[i]
            anti_ids = torch.tensor(tok(s["anti"])["input_ids"])
            stereo_ids = torch.tensor(tok(s["stereo"])["input_ids"])
            ok = all(
                find_token_range(tok, anti_ids, word, s["anti"]) is not None and
                find_token_range(tok, stereo_ids, word, s["stereo"]) is not None
                for word in s["subject"]
            )
            if ok and dds.use_empirical_blank:
                ok = s.get("anti_blank_idxs") is not None and s.get("stereo_blank_idxs") is not None
            if ok:
                la = len(tok(s["anti"])["input_ids"])
                ls = len(tok(s["stereo"])["input_ids"])
                ok = (la == ls)
            if ok:
                usable += 1
            else:
                skipped_ids.append(s["id"])
        report[dom] = {"total": len(dds), "usable": usable, "skipped_ids": skipped_ids}
        print(f"         {dom}: {usable}/{len(dds)} usable ({100.0 * usable / max(len(dds), 1):.1f}%)")
    cov_path = os.path.join(args.output_dir, "coverage.json")
    json.dump(report, open(cov_path, "w"), indent=2)
    print(f"         skipped IDs written to {cov_path}")

# ── summary ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("  ALL CHECKS PASSED")
print(f"  {args.model_source} -> {args.model_target}")
print(f"  Traced {traced} sample(s), skipped {skipped}")
print(f"  Output: {args.output_dir}/")
print(f"{'=' * 60}\n")
