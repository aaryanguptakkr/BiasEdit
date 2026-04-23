#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

N_PARALLEL=5

model_branch_pairs=(
    "EleutherAI/pythia-1b step0"
    "EleutherAI/pythia-1b step1000"
    "EleutherAI/pythia-1b step5000"
    "EleutherAI/pythia-1b step81000"
    "EleutherAI/pythia-1b step137000"
    "EleutherAI/pythia-1b step143000"
    "allenai/OLMo-2-0425-1B stage1-step10000-tokens21B"
)

domains=("gender" "profession" "race" "religion")

run_experiment() {
    local model=$1
    local branch=$2
    local domain=$3

    echo "[START] $model ($branch) - Domain: $domain"
    
    python experiments/bias_trace.py \
        --model_name="$model" \
        --branch="$branch" \
        --bias_file="data/domain/$domain.json" \
        --output_dir="results/$model/$branch/$domain/causal_trace"
        
    echo "[FINISH] $model ($branch) - Domain: $domain"
}

check_slots() {
    while [[ $(jobs -rp | wc -l) -ge $N_PARALLEL ]]; do
        # Wait for any single background process to finish
        wait -n
    done
}

for pair in "${model_branch_pairs[@]}"; do
    read -r model branch <<< "$pair"

    for domain in "${domains[@]}"; do
        check_slots
        run_experiment "$model" "$branch" "$domain" &
    done
done