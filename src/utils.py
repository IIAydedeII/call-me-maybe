import json
from typing import Any, ParamSpec
from collections.abc import Generator, Callable
from functools import wraps
from llm_sdk import Small_LLM_Model
from .models import JSONType, EnhancedPrompt

P = ParamSpec("P")


def print_generator(
    func: Callable[P, Generator[str]],
) -> Callable[P, Generator[str]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[str]:

        print("'", end="", flush=True)

        for token in func(*args, **kwargs):
            print(token, end="", flush=True)
            yield token

        print("'")

    return wrapper


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


class ModelOperator(Small_LLM_Model):

    def generate_tokens(
        self, prompt: EnhancedPrompt, is_valid: Callable[[str], bool]
    ) -> Generator[str]:
        MAX_ITERATION = 42

        input_ids: list[int] = self.encode(prompt).flatten().tolist()
        for _ in range(MAX_ITERATION):
            logits = self.get_logits_from_input_ids(input_ids)

            for token_id, _logit in enumerate(logits):
                token = self.decode([token_id])

                if not is_valid(token):
                    logits[token_id] = float("-inf")

            token_id = logits.index(max(logits))
            token = self.decode([token_id])

            if not is_valid(token):
                break

            yield token
            input_ids.append(token_id)
