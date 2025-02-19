declare -A models=(
    # [model_name]="model_path"
)

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    python main.py \
        --model_path "$model_path" \
        --model_type vllm \
        --format_type default \
        --tasks all \
        --save_name "$model_name" \
        --save_infer_texts \
        --save_infer_results \
        --config_path "config_debug.json" \
        --output_path output \
        --max_new_tokens 180 \
        --temperature 0.0 \
        --top_p 0.1 \
        --top_k 10 \

done