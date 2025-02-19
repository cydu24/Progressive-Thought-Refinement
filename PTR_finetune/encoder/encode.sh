declare -A corpus_loader_map=(

)

declare -A tokenizer_path_map=(

)

# Loop through the corpus_loader_map
for corpus_name in "${!corpus_loader_map[@]}"; do
    data_loader_info=(${corpus_loader_map[$corpus_name]})
    data_loader_path=${data_loader_info[0]}
    encode_type=${data_loader_info[1]}

    for model_name in "${!tokenizer_path_map[@]}"; do
        tokenizer_path=${tokenizer_path_map[$model_name]}

        python main.py \
            --tokenizer_path $tokenizer_path \
            --data_loader_path $data_loader_path \
            --corpus_name $corpus_name \
            --encode_type $encode_type \
            --output_path \
            --max_length 4096 \
            --merge_data \
            --save_dtype int32 \
            --tokens_per_file 1000000000
    done
done
