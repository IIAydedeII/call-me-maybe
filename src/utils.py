"""Utilities for model interaction, token generation, and value casting."""

import json
from typing import Any, ParamSpec
from collections.abc import Generator, Callable
from functools import wraps
from llm_sdk import Small_LLM_Model
from .models import JSONType, EnhancedPrompt


class AppError(Exception):
    """Base exception for application-specific errors."""


P = ParamSpec("P")


def print_generator(
    func: Callable[P, Generator[str]],
) -> Callable[P, Generator[str]]:
    """Decorate a generator to print its yielded tokens.

    Args:
        func (Callable[P, Generator[str]]):
        The generator function to decorate.

    Returns:
        Callable[P, Generator[str]]:
            A wrapped generator function that prints each yielded token before
            yielding it to the caller.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[str]:
        """Print and yield tokens produced by the decorated generator.

        Args:
            *args: Positional arguments forwarded to the original generator.
            **kwargs: Keyword arguments forwarded to the original generator.

        Yields:
            str: The tokens produced by the original generator, in order.
        """
        print("'", end="", flush=True)

        for token in func(*args, **kwargs):
            print(token, end="", flush=True)
            yield token

        print("'")

    return wrapper


class CastError(AppError):
    """Raised when the given value does not comply with the expected schema."""


def cast_value(value: str, json_type: JSONType) -> Any:
    """Convert a generated string into its expected JSON data type.

    Uses the specified JSON type to determine how the input string should
    be converted. Supports numbers, integers, booleans, and strings.

    Args:
        value (str): The string value to convert.
        json_type (JSONType): The expected JSON data type.

    Returns:
        Any: The converted value in its corresponding Python representation.

    Raises:
        CastError: If the value cannot be converted to the expected type.
    """
    try:
        match json_type:
            case JSONType.NUMBER:
                return float(value)
            case JSONType.INTEGER:
                return int(value)
            case JSONType.BOOLEAN:
                return bool(value)
            case JSONType.STRING:
                return json.loads(f'"{value}"')
    except (ValueError, json.decoder.JSONDecodeError) as e:
        raise CastError("Could not cast the given value") from e


class ModelOperator(Small_LLM_Model):
    """Extend the language model with constrained token generation.

    Inherits model encoding, decoding, and logit computation capabilities
    from Small_LLM_Model.
    """

    def generate_tokens(
        self, prompt: EnhancedPrompt, is_valid: Callable[[str], bool]
    ) -> Generator[str]:
        """Generate tokens while enforcing a custom validation constraints.

        Encodes the provided prompt and iteratively generates tokens by
        selecting the highest-scoring valid token according to the model's
        logits.

        Args:
            prompt (EnhancedPrompt):
                The enhanced prompt used as the model input.
            is_valid (Callable[[str], bool]):
                A callback that determines whether a candidate token is
                acceptable.

        Yields:
            str: Decoded token strings accepted by the validation callback.

        Note:
            The maximum number of generation iterations is 42.
        """
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

            # breakpoint()

            if not is_valid(token):
                break

            yield token
            input_ids.append(token_id)
