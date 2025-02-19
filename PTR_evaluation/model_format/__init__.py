from .default_format import format_prompt_default
from .llama3_format import format_prompt_llama3
from .qwen_format import format_prompt_qwen

MODEL_FORMAT = {
    "default": format_prompt_default,
    "llama3": format_prompt_llama3,
    "qwen": format_prompt_qwen,
}