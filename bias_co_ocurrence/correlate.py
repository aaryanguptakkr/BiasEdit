"""Correlation analysis between pre-training co-occurrence metrics and model outputs."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


def compute_pearson_r(x: Any, y: Any) -> float:
    """Compute Pearson correlation coefficient using scipy.stats.pearsonr."""
    if len(x) < 2 or len(x) != len(y) or len(set(x)) <= 1 or len(set(y)) <= 1:
        return 0.0
    return round(float(pearsonr(x, y).statistic), 4)


def compute_auroc(scores: Any, labels: Any) -> float:
    """Compute AUROC using sklearn.metrics.roc_auc_score."""
    if len(set(labels)) < 2:
        return 0.5
    return round(float(roc_auc_score(labels, scores)), 4)


def _load_domain_evals(path: Path) -> dict[str, float]:
    """Extract domain-level stereotype scores (SS) from evaluation benchmark JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: dict[str, Any] = {}
    if "stage2-ingredient3" in data and isinstance(data["stage2-ingredient3"], list):
        results = data["stage2-ingredient3"][-1].get("results", {})
    elif "stage1" in data and isinstance(data["stage1"], list):
        results = data["stage1"][-1].get("results", {})
    elif any(k.startswith("stereoset_intrasentence_") for k in data):
        results = data

    prefix = "stereoset_intrasentence_"
    return {
        k.replace(prefix, ""): float(v["ss,none"])
        for k, v in results.items()
        if k.startswith(prefix) and isinstance(v, dict) and "ss,none" in v
    }


def _load_instance_evals(path: Path) -> dict[str, dict[str, Any]]:
    """Parse instance-level stereotype predictions from a JSONL file."""
    items: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            s_ll, a_ll = entry.get("stereo_ll"), entry.get("anti_ll")
            if s_ll is not None and a_ll is not None:
                items[entry["id"]] = {
                    "delta_ll": float(s_ll) - float(a_ll),
                    "stereo_chosen": float(s_ll) > float(a_ll),
                }
    return items


def load_model_evaluations(eval_path: str | Path) -> dict[str, Any]:
    """Load model stereotype preferences from benchmark JSON or instance JSONL."""
    path = Path(eval_path)
    if path.suffix == ".json":
        return {"_type": "domain_level", "domain_ss": _load_domain_evals(path)}
    return {"_type": "instance_level", "items": _load_instance_evals(path)}


def _load_cooccurrence_df(path: Path) -> pd.DataFrame:
    """Load co-occurrence data from JSONL, JSON, or CSV into a DataFrame."""
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
    return df.dropna(subset=["delta_pmi"])


def analyze_domain_correlation(
    cooccurrence_file: Path,
    domain_ss: dict[str, float],
    eval_source: str,
) -> dict[str, Any]:
    """Correlate pre-training domain mean delta-PMI against benchmark stereotype scores."""
    df = _load_cooccurrence_df(cooccurrence_file)
    pmi_col = "delta_pmi_smoothed" if "delta_pmi_smoothed" in df.columns else "delta_pmi"

    stats_s = df.groupby("domain")[pmi_col].agg(
        n_samples="count",
        mean_delta_pmi="mean",
        pct_stereo_favored=lambda s: (s > 0).mean(),
    ).to_dict(orient="index")

    matched = [d for d in domain_ss if d in stats_s and stats_s[d]["n_samples"] > 0]
    if not matched:
        print("No matching domains found between co-occurrence file and evaluation.")
        return {}

    means_s = [stats_s[d]["mean_delta_pmi"] for d in matched]
    model_scores = [domain_ss[d] for d in matched]

    report: dict[str, Any] = {
        "type": "domain_level",
        "eval_source": eval_source,
        "domains": {
            d: {
                "n_samples": int(stats_s[d]["n_samples"]),
                "mean_delta_pmi_smoothed": round(float(stats_s[d]["mean_delta_pmi"]), 4),
                "pct_stereo_favored": round(float(stats_s[d]["pct_stereo_favored"]), 4),
                "model_ss": round(float(domain_ss[d]), 4),
            }
            for d in matched
        },
        "domain_pearson_r_smoothed": compute_pearson_r(means_s, model_scores),
    }

    if "delta_pmi_raw" in df.columns:
        df_raw = df.dropna(subset=["delta_pmi_raw"])
        if not df_raw.empty:
            stats_r = df_raw.groupby("domain")["delta_pmi_raw"].agg(
                n_samples="count", mean_delta_pmi="mean"
            ).to_dict(orient="index")
            matched_r = [d for d in domain_ss if d in stats_r and stats_r[d]["n_samples"] > 0]
            if matched_r:
                means_r = [stats_r[d]["mean_delta_pmi"] for d in matched_r]
                scores_r = [domain_ss[d] for d in matched_r]
                report["domain_pearson_r_raw"] = compute_pearson_r(means_r, scores_r)
                for d in matched_r:
                    if d in report["domains"]:
                        report["domains"][d]["mean_delta_pmi_raw"] = round(float(stats_r[d]["mean_delta_pmi"]), 4)
                        report["domains"][d]["n_samples_raw"] = int(stats_r[d]["n_samples"])

    _print_domain_report(report)
    return report


def analyze_instance_correlation(
    cooccurrence_file: Path,
    items: dict[str, Any],
) -> dict[str, Any]:
    """Correlate instance-level delta-PMI against model predictions using Pearson r and AUROC."""
    df = _load_cooccurrence_df(cooccurrence_file)
    eval_df = pd.DataFrame.from_dict(items, orient="index").reset_index(names="id")
    merged = df.merge(eval_df, on="id")

    if merged.empty:
        print("No matching instances found between co-occurrence file and evaluation file.")
        return {}

    pmi_col = "delta_pmi_smoothed" if "delta_pmi_smoothed" in merged.columns else "delta_pmi"
    report: dict[str, Any] = {
        "type": "instance_level",
        "total_matched": len(merged),
        "overall": {
            "n_samples": len(merged),
            "pearson_r_smoothed": compute_pearson_r(merged[pmi_col], merged["delta_ll"]),
            "auroc_smoothed": compute_auroc(merged[pmi_col], merged["stereo_chosen"]),
        },
        "domains": {},
    }

    if "delta_pmi_raw" in merged.columns:
        m_raw = merged.dropna(subset=["delta_pmi_raw"])
        if not m_raw.empty:
            report["overall"]["pearson_r_raw"] = compute_pearson_r(m_raw["delta_pmi_raw"], m_raw["delta_ll"])
            report["overall"]["auroc_raw"] = compute_auroc(m_raw["delta_pmi_raw"], m_raw["stereo_chosen"])
            report["overall"]["n_samples_raw"] = len(m_raw)

    for dom, group in merged.groupby("domain"):
        dom_entry = {
            "n_samples": len(group),
            "pearson_r_smoothed": compute_pearson_r(group[pmi_col], group["delta_ll"]),
            "auroc_smoothed": compute_auroc(group[pmi_col], group["stereo_chosen"]),
        }
        if "delta_pmi_raw" in group.columns:
            g_raw = group.dropna(subset=["delta_pmi_raw"])
            if not g_raw.empty:
                dom_entry["pearson_r_raw"] = compute_pearson_r(g_raw["delta_pmi_raw"], g_raw["delta_ll"])
                dom_entry["auroc_raw"] = compute_auroc(g_raw["delta_pmi_raw"], g_raw["stereo_chosen"])
                dom_entry["n_samples_raw"] = len(g_raw)
        report["domains"][dom] = dom_entry

    _print_instance_report(report)
    return report


def _save_summary_json(report: dict[str, Any], output_dir: Path) -> None:
    """Save correlation summary report to a JSON artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "correlation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_file}")


def run_correlation_analysis(
    cooccurrence_file: str | Path,
    eval_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Coordinate correlation analysis pipeline across domain or instance evals."""
    file_path, eval_p = Path(cooccurrence_file), Path(eval_path)
    model_evals = load_model_evaluations(eval_p)

    if model_evals.get("_type") == "domain_level":
        report = analyze_domain_correlation(file_path, model_evals["domain_ss"], str(eval_p))
    else:
        report = analyze_instance_correlation(file_path, model_evals.get("items", {}))

    if report and output_dir:
        _save_summary_json(report, Path(output_dir))

    return report


def _print_domain_report(report: dict[str, Any]) -> None:
    """Print domain-level correlation summary table."""
    print("\n" + "=" * 78)
    print("  OLMo 2 Training Data ΔPMI vs Model Stereotype Score (SS)")
    print(f"  Evaluation source: {report.get('eval_source', '')}")
    print("=" * 78)
    print(f"{'Domain':<15} | {'N (Smooth/Raw)':>14} | {'ΔPMI (Smooth)':>13} | {'ΔPMI (Raw)':>10} | {'Model SS':>9}")
    print("-" * 78)
    for dom, d in report["domains"].items():
        n_str = f"{d['n_samples']}/{d.get('n_samples_raw', d['n_samples'])}"
        raw_str = f"{d['mean_delta_pmi_raw']:.4f}" if "mean_delta_pmi_raw" in d else "N/A"
        print(f"{dom:<15} | {n_str:>14} | {d['mean_delta_pmi_smoothed']:>13.4f} | {raw_str:>10} | {d['model_ss']:>9.4f}")
    print("-" * 78)
    print(f"Domain Pearson r (Dirichlet Smoothed): r = {report.get('domain_pearson_r_smoothed', 0.0):.4f}")
    if "domain_pearson_r_raw" in report:
        print(f"Domain Pearson r (Raw / Unsmoothed):  r = {report['domain_pearson_r_raw']:.4f}")
    print("=" * 78)


def _print_instance_report(report: dict[str, Any]) -> None:
    """Print instance-level correlation summary table."""
    print("\n" + "=" * 70)
    print("  OLMo 2 Training Data ΔPMI vs Model Evaluation")
    print("=" * 70)
    print(f"{'Domain':<12} | {'r (Smooth)':>10} | {'r (Raw)':>8} | {'AUC (Smooth)':>12} | {'AUC (Raw)':>9}")
    print("-" * 70)
    ov = report["overall"]
    r_raw = f"{ov['pearson_r_raw']:.4f}" if "pearson_r_raw" in ov else "N/A"
    auc_raw = f"{ov['auroc_raw']:.4f}" if "auroc_raw" in ov else "N/A"
    print(f"{'OVERALL':<12} | {ov['pearson_r_smoothed']:>10.4f} | {r_raw:>8} | {ov['auroc_smoothed']:>12.4f} | {auc_raw:>9}")
    print("-" * 70)
    for dom, d in report["domains"].items():
        dr_raw = f"{d['pearson_r_raw']:.4f}" if "pearson_r_raw" in d else "N/A"
        dauc_raw = f"{d['auroc_raw']:.4f}" if "auroc_raw" in d else "N/A"
        print(f"{dom:<12} | {d['pearson_r_smoothed']:>10.4f} | {dr_raw:>8} | {d['auroc_smoothed']:>12.4f} | {dauc_raw:>9}")
    print("=" * 70)

