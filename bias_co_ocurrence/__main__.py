"""CLI entry point for pre-training co-occurrence bias analysis and model correlation."""

import argparse
from pathlib import Path

from bias_co_ocurrence.client import InfinigramClient
from bias_co_ocurrence.correlate import run_correlation_analysis
from bias_co_ocurrence.stereoset import analyze_stereoset_bias, load_stereoset_pairs

DEFAULT_OLMO1B_EVAL = "/deepfreeze/oyahia/biasolate/output_april/olmo1b/checkpoints_results.json"


def main() -> None:
    """Parse CLI arguments and run co-occurrence extraction and correlation analysis."""
    parser = argparse.ArgumentParser(description="Infini-gram StereoSet Bias & Model Correlation Analysis")
    parser.add_argument("--index", default="v4_olmo-mix-1124_llama", help="Infini-gram corpus index")
    parser.add_argument("--data-dir", default="bias_tracing/data/domain", help="Path to domain JSONs or jsonl")
    parser.add_argument("--domains", nargs="+", default=["gender", "profession", "race", "religion"])
    parser.add_argument("--output-dir", default="outputs/cooccurrence", help="Path for CSV and cache")
    parser.add_argument("--max-diff-tokens", type=int, default=50, help="Co-occurrence token window")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent worker threads")
    parser.add_argument("--eval-file", default=DEFAULT_OLMO1B_EVAL, help="Path to 1B model evaluation file (JSON/JSONL)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for dry-runs")
    args = parser.parse_args()

    client = InfinigramClient(index=args.index, cache_path=f"{args.output_dir}/cache.db")
    pairs = load_stereoset_pairs(args.data_dir, args.domains)
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Loaded {len(pairs)} pairs across domains {args.domains}.")
    print(f"Querying Infini-gram index '{args.index}' with {args.workers} workers...")
    out_file = analyze_stereoset_bias(
        client, pairs, args.output_dir, args.max_diff_tokens, max_workers=args.workers
    )
    print(f"Metrics saved to {out_file}")

    eval_path = Path(args.eval_file)
    if eval_path.exists():
        print(f"\nRunning correlation analysis against OLMo 1B evaluations at {eval_path}...")
        run_correlation_analysis(out_file, eval_path, output_dir=args.output_dir)
    else:
        print(f"\nEvaluation file {eval_path} not found; skipping correlation.")


if __name__ == "__main__":
    main()
