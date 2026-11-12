import json
from typing import Any
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
