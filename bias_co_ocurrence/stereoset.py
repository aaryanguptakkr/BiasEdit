"""StereoSet pair extraction and OLMo 2 co-occurrence bias metrics."""

import concurrent.futures
import json
import math
import string
import threading
from pathlib import Path
from typing import Any

from bias_co_ocurrence.client import InfinigramClient

# Total pre-training tokens in OLMo-mix-1124 (used for OLMo 2 1B, 7B, and 13B)
OLMO2_TOTAL_TOKENS = 4_575_475_702_047


def extract_blank_word(sentence: str, blank_idx: int) -> str:
    """Extract clean fill word from sentence at given blank token index."""
    words = sentence.split(" ")
    if blank_idx >= len(words):
        return ""
    return words[blank_idx].strip(string.punctuation).lower()


def load_stereoset_pairs(data_source: str | Path, domains: list[str]) -> list[dict[str, str]]:
    """Load target, stereo, and anti attribute pairs from JSON files or JSONL."""
    source_path = Path(data_source)
    if source_path.is_file() and str(source_path).endswith(".jsonl"):
        return _load_from_jsonl(source_path)
    return _load_from_domains(source_path, domains)


def _load_from_jsonl(file_path: Path) -> list[dict[str, str]]:
    """Parse instances from stereoset.jsonl."""
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            words = obj.get("context", "").split(" ")
            b_idx = next((i for i, w in enumerate(words) if "BLANK" in w), None)
            if b_idx is None:
                continue
            s_word = extract_blank_word(obj.get("stereo", ""), b_idx)
            a_word = extract_blank_word(obj.get("anti", ""), b_idx)
            if s_word and a_word:
                pairs.append({
                    "id": obj["id"],
                    "domain": obj.get("bias_type", "unknown"),
                    "target": obj["target"].strip().lower(),
                    "stereo_word": s_word,
                    "anti_word": a_word,
                })
    return pairs


def _load_from_domains(data_dir: Path, domains: list[str]) -> list[dict[str, str]]:
    """Parse instances from domain/*.json directory."""
    pairs = []
    for domain in domains:
        path = data_dir / f"{domain}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for obj in items:
            words = obj.get("context", "").split(" ")
            b_idx = next((i for i, w in enumerate(words) if "BLANK" in w), None)
            if b_idx is None:
                continue
            s_word = extract_blank_word(obj["data"]["stereotype"]["sentence"], b_idx)
            a_word = extract_blank_word(obj["data"]["anti-stereotype"]["sentence"], b_idx)
            if s_word and a_word:
                pairs.append({
                    "id": obj["id"],
                    "domain": domain,
                    "target": obj["target"].strip().lower(),
                    "stereo_word": s_word,
                    "anti_word": a_word,
                })
    return pairs


def compute_raw_pmi(c_joint: int, c_s: int, c_w: int, n_tokens: int = OLMO2_TOTAL_TOKENS) -> float | None:
    """Compute unsmoothed Pointwise Mutual Information (None if c_joint == 0)."""
    if c_joint <= 0 or c_s <= 0 or c_w <= 0:
        return None
    return round(math.log2((c_joint * n_tokens) / (c_s * c_w)), 4)


def compute_dirichlet_pmi(c_joint: int, c_s: int, c_w: int, n_tokens: int = OLMO2_TOTAL_TOKENS) -> float:
    """Compute Dirichlet-smoothed Pointwise Mutual Information (add-1 prior)."""
    return round(math.log2(((c_joint + 1) * n_tokens) / ((c_s + 1) * (c_w + 1))), 4)


def process_single_pair(
    item: dict[str, str], client: InfinigramClient, max_diff_tokens: int
) -> dict[str, Any]:
    """Query counts and compute both raw and Dirichlet-smoothed PMI for one pair."""
    tgt, ws, wa = item["target"], item["stereo_word"], item["anti_word"]
    c_s = client.count(tgt)
    c_ws, c_wa = client.count(ws), client.count(wa)
    c_stereo, _, _ = client.count_cooccurrence(tgt, ws, max_diff_tokens)
    c_anti, _, _ = client.count_cooccurrence(tgt, wa, max_diff_tokens)

    # Raw unsmoothed PMI
    pmi_s_raw = compute_raw_pmi(c_stereo, c_s, c_ws)
    pmi_a_raw = compute_raw_pmi(c_anti, c_s, c_wa)
    delta_pmi_raw = round(pmi_s_raw - pmi_a_raw, 4) if (pmi_s_raw is not None and pmi_a_raw is not None) else None

    # Dirichlet prior smoothed PMI (Dir(alpha=1) prior)
    pmi_s_smooth = compute_dirichlet_pmi(c_stereo, c_s, c_ws)
    pmi_a_smooth = compute_dirichlet_pmi(c_anti, c_s, c_wa)
    delta_pmi_smooth = round(pmi_s_smooth - pmi_a_smooth, 4)

    return {
        "id": item["id"],
        "domain": item["domain"],
        "target": tgt,
        "stereo_word": ws,
        "anti_word": wa,
        "count_target": c_s,
        "count_attribute_stereo": c_ws,
        "count_attribute_anti": c_wa,
        "count_joint_stereo": c_stereo,
        "count_joint_anti": c_anti,
        # Raw unsmoothed metrics (null when joint count is 0)
        "pmi_stereo_raw": pmi_s_raw,
        "pmi_anti_raw": pmi_a_raw,
        "delta_pmi_raw": delta_pmi_raw,
        # Dirichlet-smoothed metrics (add-1 prior)
        "pmi_stereo_smoothed": pmi_s_smooth,
        "pmi_anti_smoothed": pmi_a_smooth,
        "delta_pmi_smoothed": delta_pmi_smooth,
        "delta_pmi": delta_pmi_smooth,
    }


def analyze_stereoset_bias(
    client: InfinigramClient,
    pairs: list[dict[str, str]],
    output_dir: str | Path,
    max_diff_tokens: int = 50,
    max_workers: int = 6,
) -> Path:
    """Query OLMo 2 pre-training counts concurrently and stream metrics to JSONL."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_file = out_path / "olmo2_stereoset_cooccurrences.jsonl"

    existing_ids: set[str] = set()
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass

    to_process = [p for p in pairs if p["id"] not in existing_ids]
    if existing_ids:
        print(f"Resuming: found {len(existing_ids)} already processed pairs, {len(to_process)} remaining.")

    if not to_process:
        print(f"All {len(pairs)} pairs already processed in {jsonl_file}.")
        return jsonl_file

    file_lock = threading.Lock()
    completed = 0
    total = len(to_process)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_pair, item, client, max_diff_tokens)
            for item in to_process
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
            except Exception as e:
                print(f"Warning: pair processing failed: {e}", flush=True)
                continue
            with file_lock, open(jsonl_file, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(res) + "\n")
                out_f.flush()
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"Progress: {completed}/{total} new pairs processed ({completed/total*100:.1f}%)", flush=True)

    return jsonl_file
