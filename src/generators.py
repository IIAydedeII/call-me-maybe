"""Generate function names and parameter values by constraining tokens."""

from collections.abc import Generator
from .models import FunctionModel, JSONType, EnhancedPrompt
from .utils import cast_value, CastError, ModelOperator


def generate_function_name(
    model: ModelOperator,
    prompt: EnhancedPrompt,
    functions: list[FunctionModel],
) -> Generator[str]:
    """Generate a valid function name through constrained token generation.

    Generates tokens incrementally and accepts only tokens that contribute
    to a function name matching at least one of the available function
    definitions.

    Args:
        model (ModelOperator):
            The model operator responsible for generating tokens.
        prompt (EnhancedPrompt):
            The prompt used to guide function name generation.
        functions (list[FunctionModel]): The available function definitions
    Yields:
        str: Individual tokens that collectively form a valid function name.
    """

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
    """Generate a parameter value matching an expected JSON data type.

    Generates tokens incrementally while enforcing the expected JSON type.
    Token generation begins after an opening double quote is encountered
    and stops when an unescaped closing double quote is detected.

    Each candidate token is validated by attempting to cast the accumulated
    result to the expected JSON type.

    Args:
        model (ModelOperator):
            The model operator responsible for generating tokens.
        prompt (EnhancedPrompt):
            The enhanced prompt containing the parameter generation
            instructions.
        json_type (JSONType):
            The expected JSON data type of the parameter value.

    Yields:
        str: Individual tokens that collectively form the generated parameter
        value, excluding the surrounding double quotes.

    Note:
        Tokens containing double quotes are initially accepted to allow
        the generation process to identify the boundaries of the value.
        Subsequent tokens are validated against the expected JSON type.

        Generation stops when an unescaped closing double quote is found
        or when no further valid tokens can be generated.
    """
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
