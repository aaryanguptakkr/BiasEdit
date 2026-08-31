set -m

export CUDA_VISIBLE_DEVICES=0

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
    local domain=$1

    echo "[START] Olmo 1B (pre/instruct) Domain: $domain"
    
    python experiments/bias_trace.py \
        --model_source="allenai/OLMo-2-0425-1B" \
        --model_target="allenai/OLMo-2-0425-1B-Instruct" \
        --bias_file="data/domain/$domain.json" \
        --output_dir="results/cross_patch/olmo_1b_pre_to_post/$domain/causal_trace"

    python experiments/bias_trace.py \
        --model_source="allenai/OLMo-2-0425-1B-Instruct" \
        --model_target="allenai/OLMo-2-0425-1B" \
        --bias_file="data/domain/$domain.json" \
        --output_dir="results/cross_patch/olmo_1b_post_to_pre/$domain/causal_trace"
        
    echo "[FINISH] $model ($branch) - Domain: $domain"
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