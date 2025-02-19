TASK_LIST = [
    "arc",
    "commonsenseqa",
    "drop",
    "gpqa",
    "math",
    "gsm8k",
    "math",
    "gpqa"
    "hellaswag",
    "humaneval",
    "mmlu",
    "winogrande",
    "xsum",
]

from .tasks import LOAD_TASK_DATA, MATCH_TASK_ANSWER
from .model_format import MODEL_FORMAT
