#!/bin/bash
# Sanity check for the cross-model causal tracing pipeline.
# Writes ONLY to results/sanity_check/ — never touches existing results.
#
# usage: ./sanity_check.sh [source_model] [target_model] [gpu_id] [family_tag]
# defaults: OLMo-2-0425-1B -> OLMo-2-0425-1B-Instruct on GPU 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

SOURCE=${1:-allenai/OLMo-2-0425-1B}
TARGET=${2:-allenai/OLMo-2-0425-1B-Instruct}
GPU=${3:-2}
FAMILY=${4:-$(basename "$TARGET" | tr '[:upper:]' '[:lower:]')}

cd "$REPO_DIR"   # globals.yml is opened with a relative path

echo "=== Sanity Check: $SOURCE -> $TARGET ==="
echo "Repo:  $REPO_DIR"
echo "GPU:   $GPU (CUDA_VISIBLE_DEVICES=$GPU)"
echo "Env:   bias_trace_olmo"
echo ""

CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 conda run -n bias_trace_olmo \
    python "$REPO_DIR/experiments/sanity_check.py" \
    --model_source "$SOURCE" \
    --model_target "$TARGET" \
    --bias_file "$REPO_DIR/data/domain/gender.json" \
    --subject_file "$REPO_DIR/data/knowns.json" \
    --output_dir "$REPO_DIR/results/sanity_check/$FAMILY" \
    --coverage \
    --n_samples 1

echo ""
echo "=== Sanity check complete. Results written to results/sanity_check/ ==="
