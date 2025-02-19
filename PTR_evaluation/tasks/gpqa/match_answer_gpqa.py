import json
from ..match_answer import ANSWER_PATTERN_MULTICHOICE
import re

def match_answer_gpqa(infer_result:dict, round_idx:int, args):
    task_config = args.tasks_config["gpqa"]
    result = {}
    for subject in task_config["subjects"]:
        correct_cnt = 0
        for item in infer_result[subject]:
            match = re.search(ANSWER_PATTERN_MULTICHOICE, item[f"infer_round{round_idx}"])
            model_answer = match.group(1) if match else None
            item[f"judge{round_idx}"] = False
            if model_answer == item["ans"]:
                correct_cnt += 1
                item[f"judge{round_idx}"] = True
            item[f"extract_answer_round{round_idx}"] = model_answer
            
        result[subject] = {
            "acc": correct_cnt / len(infer_result[subject]),
        }

    result["gpqa"] = {
        subject: result[subject]["acc"] for subject in task_config["subjects"]
    }
    return result