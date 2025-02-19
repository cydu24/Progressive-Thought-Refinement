import os, json

def format_gpqa_query(data, has_answer):
    question_template = "{question}\n"
    prompt = "Question: " + question_template.format(
        question=data["Q"]
    ) 
    if has_answer:
        prompt += f"({data['ans']})"
    return prompt


def load_file_gpqa(fn, limit=None):
    data = []
    with open(fn, "r") as f:
        for line in f:
            question = {"Q": line["question"]}
            question["ans"] = line["answer"]
            data.append(question)
            if limit and len(data) >= limit:
                break
    return data


def load_fewshot_data(dev_data, begin_idx, num_fewshots):
    # fewshot_prompt = ""
    # for i in range(begin_idx, begin_idx + num_fewshots):
    #     fewshot_prompt += format_gpqa_query(dev_data[i % len(dev_data)], True)
    return dev_data


def load_data_gpqa(args):
    gpqa_dir = os.path.join(args.data_path, "tasks", "gpqa")
    gpqa_instruction = "The following are  questions about reasoning. "
    task_config = args.tasks_config["gpqa"]
    task_data = {}

    for subject in task_config["subjects"]:
        fn = os.path.join(gpqa_dir, "gpqa.jsonl")
        subject_data = load_file_gpqa(fn, task_config[subject]["limit"])
        cur_fewshot_begin_idx = 0
        
        task_data[subject] = []
        for item in subject_data:
            # fewshot_prompt = load_fewshot_data(cur_fewshot_begin_idx, task_config[subject]["num_fewshots"])
            # cur_fewshot_begin_idx += task_config[subject]["num_fewshots"]
            prompt = format_gpqa_query(item, False)
            task_data[subject].append({
                **item,
                "instruction": gpqa_instruction,
                "prompt_round1": prompt,
            })
    return task_data
