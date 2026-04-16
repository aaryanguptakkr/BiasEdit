#!/bin/bash
# Sanity check for OLMo-2-0425-1B causal tracing pipeline.
# Runs on GPU 2, writes ONLY to results/sanity_check/ — never touches existing results.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"   # globals.yml is opened with a relative path

echo "=== Sanity Check: OLMo-2-0425-1B Causal Tracing ==="
echo "Repo:  $REPO_DIR"
echo "GPU:   2 (CUDA_VISIBLE_DEVICES=2)"
echo "Env:   bias_trace_olmo"
echo ""

CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 conda run -n bias_trace_olmo \
    python "$REPO_DIR/experiments/sanity_check.py" \
    --model_name allenai/OLMo-2-0425-1B \
    --bias_file "$REPO_DIR/data/domain/gender.json" \
    --subject_file "$REPO_DIR/data/knowns.json" \
    --output_dir "$REPO_DIR/results/sanity_check" \
    --n_samples 3

echo ""
echo "=== Sanity check complete. Results written to results/sanity_check/ ==="
