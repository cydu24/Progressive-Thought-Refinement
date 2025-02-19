import re
from sampler.chat_completion_sampler import ChatCompletionSampler
EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()




def check_equality(equality_checker, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = equality_checker([dict(content=prompt, role="user")])
    return response.lower().strip() == "yes"



ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
def str_to_float(text: str):
    # convert string like '1,234.00' to float
    return float(text.replace(",", ""))




def match_answer_math(infer_result, round_idx, args):
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    result = {}
    for item in infer_result["math"]:
        answer = item["answer"]
        match = re.search(ANSWER_PATTERN, item[f"infer_round{round_idx}"])
        extracted_answer = match.group(1) if match else None
        score = float(check_equality(equality_checker, answer, extracted_answer))
        item[f"judge{round_idx}"] = False
        exact_match_cnt = 0
        if score:
            item[f"exact_match{round_idx}"] = extracted_answer
            exact_match_cnt += 1
            item[f"judge{round_idx}"] = True
        else:
            item[f"exact_match{round_idx}"] = None
        
    result["math"] = {
        "acc": exact_match_cnt / len(infer_result["math"]),  # exact match
    }
    return result