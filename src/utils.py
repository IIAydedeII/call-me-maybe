import json
from typing import Any
from collections.abc import Generator, Callable
from llm_sdk import Small_LLM_Model
from .models import JSONType


def build_prompt(system: str, user: str, assistant: str = "") -> str:
    return "\n".join(
        [
            "<|im_start|>system",
            f"{system}<|im_end|>",
            "<|im_start|>user",
            f"{user}<|im_end|>",
            "<|im_start|>assistant",
            f"{assistant}",
        ]
    )


def cast_value(value: str, json_type: JSONType) -> Any:
    match json_type:
        case JSONType.NUMBER:
            return float(value)
        case JSONType.INTEGER:
            return int(value)
        case JSONType.BOOLEAN:
            return bool(value)
        case JSONType.STRING:
            return json.loads(f'"{value}"')


model = Small_LLM_Model()


def generate_tokens(
    prompt: str, is_valid: Callable[[str], bool]
) -> Generator[str]:
    MAX_ITERATION = 42

    input_ids: list[int] = model.encode(prompt).flatten().tolist()
    for _ in range(MAX_ITERATION):
        logits = model.get_logits_from_input_ids(input_ids)

        for token_id, _logit in enumerate(logits):
            token = model.decode([token_id])

            if not is_valid(token):
                logits[token_id] = float("-inf")

        token_id = logits.index(max(logits))
        token = model.decode([token_id])

        if not is_valid(token):
            break

        yield token
        input_ids.append(token_id)
