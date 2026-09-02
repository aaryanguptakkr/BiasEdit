#!/bin/bash
# Cross-model (base <-> instruct) causal tracing over the bias domains (gender,
# profession, race), both directions.
# Generalized from the original OLMo-only script — same structure, models/GPU as parameters.
#
# usage: ./cross_patch.sh <base_model> <instruct_model> <family_tag> <gpu_id> [base_revision] [instruct_revision]
# e.g.:  ./cross_patch.sh allenai/OLMo-2-0425-1B allenai/OLMo-2-0425-1B-Instruct olmo_1b 0
#        ./cross_patch.sh Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct qwen2.5_1.5b 1
#        ./cross_patch.sh meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-1B-Instruct llama3.2_1b 2
#        ./cross_patch.sh google/gemma-3-1b-pt google/gemma-3-1b-it gemma3_1b 3
# optional [base_revision]/[instruct_revision] are HF branch names (e.g. OLMo
# stage1-step10000-tokens21B) forwarded to --branch1/--branch2 for checkpoint-level runs.

set -m

# Results go to the shared root, not into a checkout, so they outlive any one clone
# and plotting finds them without knowing whose working copy produced them. The path
# is read from private_names.py (gitignored) — the same value plot_utils reads, so the
# writer and the reader cannot drift apart. Override with RESULTS_ROOT=... if needed.
if [[ -z "$RESULTS_ROOT" ]]; then
    RESULTS_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && python -c \
        "from private_names import shared_results_root as r; print(r)" 2>/dev/null)
fi
if [[ -z "$RESULTS_ROOT" ]]; then
    echo "cannot determine RESULTS_ROOT: set it, or define shared_results_root in private_names.py" >&2
    exit 1
fi
echo "Results root: $RESULTS_ROOT"

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "usage: $0 <base_model> <instruct_model> <family_tag> <gpu_id> [base_revision] [instruct_revision]"
    exit 1
fi

BASE=$1
INSTRUCT=$2
FAMILY=$3
export CUDA_VISIBLE_DEVICES=$4
BASE_REV=${5:-}
INSTRUCT_REV=${6:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"   # bias_tracing/ — globals.yml is opened with a relative path

N_PARALLEL=5

domains=("gender" "profession" "race")

run_experiment() {
    local domain=$1

    echo "[START] $FAMILY (pre/instruct) Domain: $domain"

    python experiments/bias_trace.py \
        --model_source="$BASE" \
        --model_target="$INSTRUCT" \
        ${BASE_REV:+--branch1="$BASE_REV"} \
        ${INSTRUCT_REV:+--branch2="$INSTRUCT_REV"} \
        --bias_file="data/domain/$domain.json" \
        --output_dir="$RESULTS_ROOT/cross_patch/${FAMILY}_pre_to_post/$domain/causal_trace"

    python experiments/bias_trace.py \
        --model_source="$INSTRUCT" \
        --model_target="$BASE" \
        ${INSTRUCT_REV:+--branch1="$INSTRUCT_REV"} \
        ${BASE_REV:+--branch2="$BASE_REV"} \
        --bias_file="data/domain/$domain.json" \
        --output_dir="$RESULTS_ROOT/cross_patch/${FAMILY}_post_to_pre/$domain/causal_trace"

    echo "[FINISH] $FAMILY - Domain: $domain"
}

check_slots() {
    while [[ $(jobs -rp | wc -l) -ge $N_PARALLEL ]]; do
        # Wait for any single background process to finish
        wait -n
    done
}

for domain in "${domains[@]}"; do
    check_slots
    run_experiment "$domain" &
done

wait
