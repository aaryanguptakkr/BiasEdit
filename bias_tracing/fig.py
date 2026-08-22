import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


SCORE_METRIC = "blank_logprob_stereo_minus_anti_v1"


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True,
                    help='the path of causal_trace/cases')
parser.add_argument("--num_layer", type=int, default=None,
                    help="optional expected number of model layers")
parser.add_argument("--model_name", type=str, default=None,
                    help="optional display name; source and target are read from result metadata")
parser.add_argument("--bias", type=str,
                    choices=["gender", "race", "profession", "religion"],
                    default="gender")
parser.add_argument("--num_sample", type=int, default=500,
                    help="maximum number of matched cases")
parser.add_argument("--output_dir", type=str, default="results",
                    help="directory for generated figures")
args = parser.parse_args()


def scalar_text(value):
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError(f"Expected scalar metadata, got shape {value.shape}")
    value = value.item()
    return value.decode() if isinstance(value, bytes) else str(value)


def split_result_files(root):
    result_files = {"single": {}, "mlp": {}, "attn": {}}
    for filename in sorted(os.listdir(root)):
        if not filename.endswith(".npz"):
            continue
        stem = filename[:-4]
        if stem.endswith("_attn"):
            kind, case_id = "attn", stem[:-5]
        elif stem.endswith("_mlp"):
            kind, case_id = "mlp", stem[:-4]
        else:
            kind, case_id = "single", stem
        result_files[kind][case_id] = os.path.join(root, filename)
    return result_files


def load_result(path, expected_kind):
    with np.load(path, allow_pickle=True) as data:
        result = {key: data[key] for key in data.files}

    required = {
        "scores", "score_metric", "high_score", "low_score",
        "source_model", "target_model", "direction", "num_layers",
        "corrupt_range_anti", "blank_idxs_anti",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    if scalar_text(result["score_metric"]) != SCORE_METRIC:
        raise ValueError(f"{path} uses an old score metric; rerun bias_trace.py")

    stored_kind = scalar_text(result.get("kind", ""))
    expected_value = "" if expected_kind == "single" else expected_kind
    if stored_kind != expected_value:
        raise ValueError(f"{path} stores kind={stored_kind!r}, expected {expected_value!r}")

    scores = np.asarray(result["scores"], dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError(f"{path} has invalid scores with shape {scores.shape}")
    if int(np.asarray(result["num_layers"]).item()) != scores.shape[1]:
        raise ValueError(f"{path} layer metadata does not match scores")

    source_model = scalar_text(result["source_model"])
    target_model = scalar_text(result["target_model"])
    if scalar_text(result["direction"]) != f"{source_model} -> {target_model}":
        raise ValueError(f"{path} has inconsistent source/target direction metadata")

    high_score = float(np.asarray(result["high_score"]).item())
    low_score = float(np.asarray(result["low_score"]).item())
    gap = high_score - low_score
    if not np.isfinite(gap) or abs(gap) < 1e-8:
        return None

    nie = (scores - low_score) / gap
    if "nie" in result and not np.allclose(nie, result["nie"], equal_nan=True):
        raise ValueError(f"{path} stores NIE inconsistent with high/low scores")

    result["scores"] = scores
    result["nie"] = nie
    result["source_model"] = source_model
    result["target_model"] = target_model
    result["high_score"] = high_score
    result["low_score"] = low_score
    return result


def mean_ranges(result, key):
    rows = []
    for begin, end in np.asarray(result[key], dtype=int).reshape(-1, 2):
        if not 0 <= begin < end <= result["nie"].shape[0]:
            raise ValueError(f"Invalid token range {(begin, end)}")
        rows.append(result["nie"][begin:end])
    if not rows:
        return None
    return np.concatenate(rows, axis=0).mean(axis=0)


def mean_blank(result):
    begin, end = np.asarray(result["blank_idxs_anti"], dtype=int).tolist()
    if not 0 <= begin < end <= result["nie"].shape[0]:
        raise ValueError(f"Invalid BLANK range {(begin, end)}")
    return result["nie"][begin:end].mean(axis=0)


def token_before_blank(result):
    begin = int(np.asarray(result["blank_idxs_anti"])[0])
    if begin == 0:
        return None
    return result["nie"][begin - 1]


def aggregate(cases, selector):
    case_means = [value for result in cases if (value := selector(result)) is not None]
    if not case_means:
        raise ValueError("No cases remain for this aggregation")
    return np.stack(case_means).mean(axis=0)


def save_bars(series, labels, title, output_path):
    num_layers = len(series[0])
    layers = np.arange(num_layers)
    bar_width = 0.25
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["blue", "green", "red"]
    for values, label, color, offset in zip(series, labels, colors, offsets):
        ax.bar(layers + offset, values, color=color, width=bar_width,
               edgecolor="gray", label=label)

    tick_step = max(1, num_layers // 8)
    ax.set_xticks(np.arange(0, num_layers, tick_step))
    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_ylabel("Normalized indirect effect (NIE)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)


result_files = split_result_files(args.root)
case_ids = set(result_files["single"])
case_ids &= set(result_files["mlp"])
case_ids &= set(result_files["attn"])
case_ids = sorted(case_ids)[:args.num_sample]
if not case_ids:
    raise ValueError(f"No matched single/MLP/attention result files found in {args.root}")

cases = {"single": [], "mlp": [], "attn": []}
skipped_gap = 0
for case_id in tqdm(case_ids):
    loaded = {
        kind: load_result(result_files[kind][case_id], kind)
        for kind in cases
    }
    if any(result is None for result in loaded.values()):
        skipped_gap += 1
        continue

    baselines = [(result["high_score"], result["low_score"]) for result in loaded.values()]
    if not all(np.allclose(baselines[0], baseline) for baseline in baselines[1:]):
        raise ValueError(f"Baseline scores differ across variants for {case_id}")
    for kind, result in loaded.items():
        cases[kind].append(result)

if not cases["single"]:
    raise ValueError("All matched cases have zero or invalid clean-corrupted gaps")

directions = {
    (result["source_model"], result["target_model"])
    for kind_cases in cases.values()
    for result in kind_cases
}
if len(directions) != 1:
    raise ValueError(f"Mixed source/target directions in {args.root}: {sorted(directions)}")
source_model, target_model = directions.pop()

num_layers = {result["nie"].shape[1] for kind_cases in cases.values() for result in kind_cases}
if len(num_layers) != 1:
    raise ValueError(f"Mixed layer counts in {args.root}: {sorted(num_layers)}")
num_layers = num_layers.pop()
if args.num_layer is not None and args.num_layer != num_layers:
    raise ValueError(f"Expected {args.num_layer} layers, result files contain {num_layers}")

single_subject = aggregate(cases["single"], lambda result: mean_ranges(result, "corrupt_range_anti"))
mlp_subject = aggregate(cases["mlp"], lambda result: mean_ranges(result, "corrupt_range_anti"))
attn_subject = aggregate(cases["attn"], lambda result: mean_ranges(result, "corrupt_range_anti"))
pre_blank = aggregate(cases["single"], token_before_blank)
blank = aggregate(cases["single"], mean_blank)

direction_name = f"{source_model} → {target_model}"
display_name = f"{args.model_name} ({direction_name})" if args.model_name else direction_name
run_name = f"{source_model}_to_{target_model}".replace("/", "_").replace("\\", "_")
os.makedirs(args.output_dir, exist_ok=True)

states_path = os.path.join(args.output_dir, f"{run_name}-{args.bias}-states.png")
save_bars(
    [single_subject, mlp_subject, attn_subject],
    ["Single-state restoration", "MLP-window restoration", "Attention-window restoration"],
    f"{args.bias.title()} bias effect of subject states ({display_name})",
    states_path,
)

words_path = os.path.join(args.output_dir, f"{run_name}-{args.bias}-words.png")
save_bars(
    [single_subject, pre_blank, blank],
    ["Subject token", "Token before target", "Target token"],
    f"{args.bias.title()} bias effect by token position ({display_name})",
    words_path,
)

print(f"Plotted {len(cases['single'])} cases ({skipped_gap} zero-gap cases skipped)")
print(states_path)
print(words_path)
