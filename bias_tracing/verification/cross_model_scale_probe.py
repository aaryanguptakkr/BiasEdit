"""Cross-model restoration scale probe for experiments/bias_trace.py.

Question: does injecting the SOURCE model's clean activations into the corrupted
TARGET run move the target's stereo-anti gap toward the target's CLEAN gap
(the value the NIE normalizes against), or somewhere else?

For each sentence and each direction we measure four whole-sentence gaps:
  high      = target clean                        (NIE denominator top)
  low       = target, subject embeddings noised   (NIE denominator bottom)
  self_all  = target corrupted, then ALL (token,layer) residuals restored
              from the target's own clean forward   -> must equal `high` (ROME identity)
  cross_all = same, restored from the SOURCE model  -> no a-priori value
and the implied full-restore NIE  (cross_all - low) / (high - low), which is 1.0
by construction for a within-model trace and should be ~1 if the cross-model
activation basis is shared.

CPU-friendly (small batch, whole-sentence score only, no per-cell loop).
Run from bias_tracing/ :
  PYTHONNOUSERSITE=1 conda run --no-capture-output -n bias_trace_olmo \
    python verification/cross_model_scale_probe.py --family olmo --n 30
"""
import argparse, json, os, sys, statistics as st
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.append("./")
sys.path.append("experiments")

import torch
import bias_trace as bt
from dsets import StereoSetDataset

FAMILIES = {
    "olmo":  ("allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-Instruct"),
    "qwen":  ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "llama": ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),
    "gemma": ("google/gemma-3-1b-pt", "google/gemma-3-1b-it"),
}


def load(name, device, dtype):
    mt = bt.ModelAndTokenizer(name, torch_dtype=dtype)
    mt.model.to(device)
    return mt


def gaps_for_direction(mt_src, mt_tgt, cases, noise_factor, device, samples):
    """Yield dict(high, low, self_all, cross_all) per usable case."""
    subj_words = None
    ds_cache = {}
    L = mt_tgt.num_layers
    layers = [bt.layername(mt_tgt.model, l) for l in range(L)]
    noise = noise_factor * bt.collect_embedding_std(
        mt_tgt, [c["subject"][0] for c in cases[:64]], device=device)

    out = []
    for c in cases:
        made = {}
        for side in ("anti", "stereo"):
            r = bt.make_inputs(mt_tgt, prompts=[c[side]] * (samples + 1),
                               labels=[c[side]] * (samples + 1),
                               subject=c["subject"], blank_idxs=c[f"{side}_blank_idxs"],
                               device=device)
            if r[0] is None:
                break
            made[side] = r
        if len(made) != 2:
            continue
        (ia, ea, _, _), (is_, es_, _, _) = made["anti"], made["stereo"]
        if ia["input_ids"].shape[1] != is_["input_ids"].shape[1]:
            continue
        sla = bt.sentence_labels(ia, mt_tgt.tokenizer.pad_token_id)
        sls = bt.sentence_labels(is_, mt_tgt.tokenizer.pad_token_id)

        def gap(oa, os_, rows=slice(None)):
            return bt.causal_difference(bt._logits(oa)[rows], sla[rows],
                                        bt._logits(os_)[rows], sls[rows]).item()

        with torch.no_grad():
            high = gap(mt_tgt.model(**ia), mt_tgt.model(**is_))
        low = gap(
            bt.trace_with_patch(mt_src.model, mt_tgt.model, ia, [], ea, noise=noise),
            bt.trace_with_patch(mt_src.model, mt_tgt.model, is_, [], es_, noise=noise),
            rows=slice(1, None))

        allst_a = [(t, ln) for t in range(ia["input_ids"].shape[1]) for ln in layers]
        allst_s = [(t, ln) for t in range(is_["input_ids"].shape[1]) for ln in layers]
        sc_self_a = bt.trace_source_states(mt_tgt.model, ia, layers)
        sc_self_s = bt.trace_source_states(mt_tgt.model, is_, layers)
        self_all = gap(
            bt.trace_with_patch(mt_tgt.model, mt_tgt.model, ia, allst_a, ea, noise=noise, source_cache=sc_self_a),
            bt.trace_with_patch(mt_tgt.model, mt_tgt.model, is_, allst_s, es_, noise=noise, source_cache=sc_self_s),
            rows=slice(1, None))
        sc_x_a = bt.trace_source_states(mt_src.model, ia, layers)
        sc_x_s = bt.trace_source_states(mt_src.model, is_, layers)
        cross_all = gap(
            bt.trace_with_patch(mt_src.model, mt_tgt.model, ia, allst_a, ea, noise=noise, source_cache=sc_x_a),
            bt.trace_with_patch(mt_src.model, mt_tgt.model, is_, allst_s, es_, noise=noise, source_cache=sc_x_s),
            rows=slice(1, None))
        out.append(dict(id=c["id"], high=high, low=low, self_all=self_all, cross_all=cross_all))
        print(f"    case {len(out):3d}  high={high:+.4f} low={low:+.4f} "
              f"self_all={self_all:+.4f} cross_all={cross_all:+.4f}", flush=True)
    return out


def summarize(tag, rows):
    def med(k):
        return st.median([r[k] for r in rows])
    self_err = [abs(r["self_all"] - r["high"]) for r in rows]
    nie_full = []
    for r in rows:
        g = r["high"] - r["low"]
        if abs(g) > 1e-6:
            nie_full.append((r["cross_all"] - r["low"]) / g)
    print(f"\n[{tag}]  n={len(rows)}")
    print(f"  median  high={med('high'):+.4f}  low={med('low'):+.4f}  "
          f"self_all={med('self_all'):+.4f}  cross_all={med('cross_all'):+.4f}")
    print(f"  self-restore identity error (max)  = {max(self_err):.2e}   (want ~0)")
    if nie_full:
        nie_full.sort()
        print(f"  implied full-restore NIE  median={st.median(nie_full):+.2f}  "
              f"IQR=[{nie_full[len(nie_full)//4]:+.2f}, {nie_full[3*len(nie_full)//4]:+.2f}]   (want ~+1.00)")
    frac_between = sum(min(r['high'], r['low']) <= r['cross_all'] <= max(r['high'], r['low'])
                       for r in rows) / len(rows)
    print(f"  cross_all lands between low and high in {100*frac_between:.0f}% of cases   (want ~100%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), default="olmo")
    ap.add_argument("--bias_file", default="data/domain/gender.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base_name, inst_name = FAMILIES[a.family]
    dtype = getattr(torch, a.dtype)
    print(f"loading {base_name} + {inst_name}  ({a.dtype}, {a.device})")
    mt_base = load(base_name, a.device, dtype)
    mt_inst = load(inst_name, a.device, dtype)
    bt.validate_model_pair(mt_base, mt_inst)

    ds = StereoSetDataset(mt_inst.tokenizer, a.bias_file, inst_name)
    cases = [ds[i] for i in range(len(ds))]
    cases = [c for c in cases if c["anti_blank_idxs"] and c["stereo_blank_idxs"]][: a.n * 2]

    result = {}
    print("\n=== direction pre_to_post: source=base, target=instruct ===")
    r1 = gaps_for_direction(mt_base, mt_inst, cases, a.noise, a.device, a.samples)[: a.n]
    summarize("pre_to_post", r1)
    print("\n=== direction post_to_pre: source=instruct, target=base ===")
    r2 = gaps_for_direction(mt_inst, mt_base, cases, a.noise, a.device, a.samples)[: a.n]
    summarize("post_to_pre", r2)

    if a.out:
        json.dump({"family": a.family, "pre_to_post": r1, "post_to_pre": r2}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
