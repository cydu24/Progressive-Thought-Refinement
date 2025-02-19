#!/bin/bash

# Set default values for the arguments
MODEL_PATH="path/to/your/model"
MODEL_NAME="your_model_name"
OUTPUT_PATH="output"
USE_VLLM=false
USE_CPU=false
TEMPERATURE=1.0
TOP_P=1
TOP_K=-1
PROMPT_PATHS=("path/to/prompt1.jsonl" "path/to/prompt2.jsonl")
DATA_LEN=10000

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --use_vllm) USE_VLLM=true ;;
        --use_cpu) USE_CPU=true ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --top_p) TOP_P="$2"; shift ;;
        --top_k) TOP_K="$2"; shift ;;
        --prompt_paths) IFS=',' read -r -a PROMPT_PATHS <<< "$2"; shift ;;
        --data_len) DATA_LEN="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the specified arguments
python3 thought_generation_vllm.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --output_path "$OUTPUT_PATH" \
    $( [ "$USE_VLLM" = true ] && echo "--use_vllm" ) \
    $( [ "$USE_CPU" = true ] && echo "--use_cpu" ) \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --prompt_paths "${PROMPT_PATHS[@]}" \
    --data_len "$DATA_LEN"