"""
Sanity check for the OLMo-2-0425-1B causal tracing pipeline.
Runs each stage independently and reports PASS/FAIL.
Writes ONLY to --output_dir (default: results/sanity_check/).
Never touches existing causal trace results.
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # bias_tracing/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                      # experiments/

# ── helpers ────────────────────────────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}")
        return result
    except Exception as e:
        print(f"  {FAIL} {label}")
        traceback.print_exc()
        sys.exit(1)

# ── args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",   default="allenai/OLMo-2-0425-1B")
parser.add_argument("--bias_file",    default="data/domain/gender.json")
parser.add_argument("--subject_file", default="data/knowns.json")
parser.add_argument("--output_dir",   default="results/sanity_check")
parser.add_argument("--n_samples",    default=3, type=int,
                    help="Number of dataset samples to trace (keep small)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── stage 1: imports ───────────────────────────────────────────────────────────

print("\n[1/6] Checking imports ...")

check("numpy / torch / matplotlib",
      lambda: __import__("numpy") and __import__("torch") and __import__("matplotlib"))

check("transformers (AutoModelForCausalLM, AutoTokenizer)",
      lambda: getattr(__import__("transformers"), "AutoModelForCausalLM"))

check("dsets.StereoSetDataset",
      lambda: getattr(__import__("dsets", fromlist=["StereoSetDataset"]), "StereoSetDataset"))

check("util.nethook",
      lambda: __import__("util.nethook", fromlist=["nethook"]))

# ── stage 2: CUDA ─────────────────────────────────────────────────────────────

print("\n[2/6] Checking CUDA ...")

import torch

def _cuda_check():
    assert torch.cuda.is_available(), "CUDA not available"
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    print(f"         device count={n}  device[0]={name}")
    return True

check("CUDA available + device info", _cuda_check)

# ── stage 3: model + tokenizer ────────────────────────────────────────────────

print(f"\n[3/6] Loading ModelAndTokenizer ({args.model_name}) ...")

from bias_trace import ModelAndTokenizer, layername

def _load_model():
    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch.float16)
    print(f"         iscausal={mt.iscausal}  num_layers={mt.num_layers}")
    assert mt.iscausal, "OLMo should be treated as causal LM"
    assert mt.num_layers > 0, "num_layers must be > 0"
    # verify layername resolution works
    l0 = layername(mt.model, 0)
    le = layername(mt.model, 0, "embed")
    la = layername(mt.model, 0, "attn")
    lm = layername(mt.model, 0, "mlp")
    print(f"         layer[0]={l0}  embed={le}  attn={la}  mlp={lm}")
    return mt

mt = check("ModelAndTokenizer loads + iscausal=True", _load_model)

# ── stage 4: dataset ──────────────────────────────────────────────────────────

print(f"\n[4/6] Loading StereoSetDataset ({args.bias_file}) ...")

from dsets import StereoSetDataset

def _load_dataset():
    ds = StereoSetDataset(mt.tokenizer, args.bias_file, args.model_name)
    assert ds.isolmo, "isolmo flag must be True for OLMo model"
    assert len(ds) > 0, "Dataset must have at least one sample"
    sample = ds[0]
    expected_keys = {"id", "anti", "stereo", "anti_mask", "stereo_mask", "subject"}
    missing = expected_keys - set(sample.keys())
    assert not missing, f"Sample missing keys: {missing}"
    print(f"         isolmo={ds.isolmo}  n_samples={len(ds)}")
    print(f"         sample subject={sample['subject']}")
    return ds

ds = check("StereoSetDataset loads with isolmo=True", _load_dataset)

# ── stage 5: make_inputs + noise level ────────────────────────────────────────

print("\n[5/6] Checking make_inputs and noise level ...")

import json
from bias_trace import make_inputs, collect_embedding_std

def _make_inputs_check():
    sample = ds[0]
    inp_anti, e_range_anti, blank_idxs_anti, inp_anti_origin = make_inputs(
        mt,
        prompts=[sample['anti']] * 2,
        labels=[sample['anti_mask']] * 2,
        subject=sample['subject']
    )
    assert inp_anti is not None, "make_inputs returned None for anti"
    assert "input_ids" in inp_anti, "make_inputs missing input_ids"
    assert "labels" in inp_anti, "make_inputs missing labels"
    print(f"         input shape={tuple(inp_anti['input_ids'].shape)}  "
          f"e_range={e_range_anti}  blank_idxs={blank_idxs_anti}")
    return inp_anti, e_range_anti, blank_idxs_anti

inp_anti, e_range_anti, blank_idxs_anti = check("make_inputs for OLMo sample", _make_inputs_check)

def _noise_check():
    subjects = json.load(open(args.subject_file))
    noise_level = 3.0 * collect_embedding_std(mt, subjects[:20])
    print(f"         noise_level={noise_level:.6f}  (computed over 20 subjects)")
    assert noise_level > 0, "noise_level must be positive"
    return noise_level

noise_level = check("collect_embedding_std (20 subjects)", _noise_check)

# ── stage 6: end-to-end trace on n_samples ────────────────────────────────────

print(f"\n[6/6] End-to-end causal trace ({args.n_samples} samples) ...")
print(f"       Output dir: {args.output_dir}")

import numpy
from bias_trace import calculate_hidden_flow

traced = 0
skipped = 0

for knowledge in ds:
    if traced >= args.n_samples:
        break

    # skip samples where subject word not in both sentences
    if any(w not in knowledge['anti'] or w not in knowledge['stereo']
           for w in knowledge['subject']):
        skipped += 1
        continue

    inp_a, e_a, bi_a, inp_a_o = make_inputs(
        mt,
        prompts=[knowledge['anti']] * 2,
        labels=[knowledge['anti_mask']] * 2,
        subject=knowledge['subject']
    )
    inp_s, e_s, bi_s, inp_s_o = make_inputs(
        mt,
        prompts=[knowledge['stereo']] * 2,
        labels=[knowledge['stereo_mask']] * 2,
        subject=knowledge['subject']
    )
    if inp_a is None or inp_s is None:
        skipped += 1
        continue
    if inp_a["input_ids"].shape[1] != inp_s["input_ids"].shape[1]:
        skipped += 1
        continue

    result = calculate_hidden_flow(
        mt, knowledge,
        inp_a, inp_s,
        e_a, e_s,
        bi_a, bi_s,
        inp_a_o, inp_s_o,
        noise=noise_level,
        kind=None,
    )
    if not result:
        skipped += 1
        continue

    out_path = os.path.join(args.output_dir, f"sanity_{knowledge['id']}.npz")
    numpy.savez(out_path, **{
        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
        for k, v in result.items()
    })
    traced += 1
    print(f"  {PASS} Traced sample {traced}/{args.n_samples}: {knowledge['id']}  "
          f"→ {out_path}")

if traced == 0:
    print(f"  {FAIL} No samples were successfully traced (skipped={skipped})")
    sys.exit(1)

# ── summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  ALL CHECKS PASSED")
print(f"  Traced {traced} sample(s), skipped {skipped}")
print(f"  Output: {args.output_dir}/")
print(f"{'='*60}\n")
