from collections.abc import Generator
from .models import FunctionModel, JSONType, EnhancedPrompt
from .utils import cast_value, CastError, ModelOperator


def generate_function_name(
    model: ModelOperator,
    prompt: EnhancedPrompt,
    functions: list[FunctionModel],
) -> Generator[str]:
    result = ""

    def is_valid(token: str) -> bool:
        return bool(token) and any(
            fn.name.startswith(result + token) for fn in functions
        )

    for token in model.generate_tokens(prompt, is_valid):
        result += token
        yield token


def generate_parameter(
    model: ModelOperator, prompt: EnhancedPrompt, json_type: JSONType
) -> Generator[str]:
    result = ""
    quote_open = False

    def is_valid(token: str) -> bool:
        if not token:
            return False

        # Allow all tokens with a double quote in them
        if '"' in token:
            return True

        # Ensure the first token contains only a double quote
        if not quote_open:
            return False

        try:
            cast_value(result + token, json_type)
            return True
        except CastError:
            return False

    for token in model.generate_tokens(prompt, is_valid):
        if not quote_open and '"' in token:
            quote_open = True

            # Remove the part before the opening quote
            token = token.split('"', 1)[1]

        if quote_open and '"' in token and '\\"' not in token:
            # Remove the part after the closing quote
            yield token.split('"', 1)[0]
            break

        result += token
        yield token
