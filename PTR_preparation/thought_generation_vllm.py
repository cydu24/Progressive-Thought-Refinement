from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
from tqdm import tqdm
import datetime
import torch
vllm_initialized = False

def vllm_initialize(args):
    global vllm_initialized
    global sampling_params
    global model
    model = LLM(
        model=args.model_path, 
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    sampling_params = SamplingParams(
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop=['<|im_end|>','<|endoftext|>','<|eot_id|>', '<|/s|>'],
        temperature=args.temperature,
    )
    vllm_initialized = True

def vllm_generate(prompts, args, generation_name=""):
    global vllm_initialized
    global model
    global sampling_params
    if not vllm_initialized:
        vllm_initialize(args)
    generated_texts = []
    prompts_list = []
    prompts_per_step = 1000
    for i in range(0, len(prompts), prompts_per_step):
        prompts_list.append(prompts[i:i+prompts_per_step])

    for i, inputs in enumerate(prompts_list):
        print(f"generating {generation_name} [{i * prompts_per_step + 1}, {len(prompts)}] in {len(prompts)} prompts")
        outputs = model.generate(inputs, sampling_params)
        for i in range(len(outputs)):
            output = outputs[i]
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text.strip())
    return generated_texts


def format_example(line):
    data = json.loads(line) 
    question = data["question"]
    example = "Question: " + question
    example += "\nAnswer:"
    return example.strip(),question.strip()

# output question
def load_prompts(args):
    prompts = []
    questions = []
    for fn in args.prompt_paths:
        with open(fn, "r") as f:
            for line in f:
                inputs, question = format_example(line)
                # print(inputs)
                questions.append(question)
                prompts.append(inputs)

    return prompts,questions


def load_model(args):
    print("loading model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    return model, tokenizer


def hf_generate(prompts, args):
    model, tokenizer = load_model(args)
    device = torch.device("cpu" if args.use_cpu else "cuda")
    outputs = []
    model.to(device)
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_ids = model.generate(
            inputs.input_ids.to(device), 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        outputs.append(output.strip())
    return outputs


def output(result, args):
    model_name = args.model_name
    os.makedirs(args.output_path, exist_ok=True)
    t = datetime.datetime.now()
    output_file_name = os.path.join(args.output_path, f"{t.month}-{t.day}_{t.hour:02d}:{t.minute:02d}_{model_name}_len_{args.data_len}.json")
    with open(output_file_name, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--prompt_paths", nargs='+')
    parser.add_argument("--data_len", type=int, default = 10000)
    args = parser.parse_args()   
    if args.use_cpu:
        args.max_new_tokens = 128
    return args 


if __name__ == "__main__":
    args = parse_args()
    
    prompts, questions = load_prompts(args)
    prompts = prompts[:args.data_len]
    result = []
    if args.use_vllm:
        outputs = vllm_generate(prompts, args)
    else:
        outputs = hf_generate(prompts, args)
    for i in range(len(prompts)):
        result.append({
            "question":questions[i],
            "input":prompts[i],
            "output":outputs[i]
        })
    output(result, args)