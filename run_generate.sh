
declare -a model_paths=(

)

declare -a model_names=(

)

for i in "${!model_paths[@]}"; do
    echo "Downloading ${model_names[$i]}..."
    python PTR_preparation/thought_generation_vllm.py \
    --model_path "${model_paths[$i]}" \
    --model_name "${model_names[$i]}" \
    --max_new_tokens 2048 \
    --prompt_paths  \
    --temperature 0.95 \
    --top_p 0.9 \
    --top_k 30 \
    --use_vllm \
    --data_len 
done


echo "All models have been downloaded successfully."
