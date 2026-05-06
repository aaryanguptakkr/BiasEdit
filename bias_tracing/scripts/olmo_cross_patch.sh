domains=("gender" "profession" "race" "religion")

for domain in "${domains[@]}"; do
    python experiments/bias_trace.py \
        --bias_file="data/domain/$domain.json" \
        --output_dir="results/cross_patch/olmo_1b_pre_olmo_1b_post_base/$domain/causal_trace"
done