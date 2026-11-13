import sys
import json
from collections.abc import Generator
from typing import Any
from .arguments import build_parser
from .input import input_model_list, InputError
from .models import (
    FunctionModel,
    PromptModel,
    JSONType,
    FunctionCall,
    EnhancedPrompt,
)
from .utils import print_generator, cast_value, ModelOperator
from .output import output_model_list, OutputError


@print_generator
def generate_function_name(
    model: ModelOperator,
    prompt: EnhancedPrompt,
    functions: list[FunctionModel],
) -> Generator[str]:
    result = ""

    function_names = [fn.name for fn in functions]

    def is_valid(token: str) -> bool:

        return bool(token) and any(
            name.startswith(result + token) for name in function_names
        )

    for token in model.generate_tokens(prompt, is_valid):
        result += token
        yield token


@print_generator
def generate_parameter(
    model: ModelOperator, prompt: EnhancedPrompt, json_type: JSONType
) -> Generator[str]:
    result = ""
    quote_open = False

    def is_valid(token: str) -> bool:
        # Allow/ignore all tokens with a double quote in them
        if not token or '"' in token:
            return True

        # Ensure the first token contains a double quote
        if not quote_open and '"' not in token:
            return False

        try:
            cast_value(result + token, json_type)
            return True
        except (ValueError, json.decoder.JSONDecodeError):
            return False

    for token in model.generate_tokens(prompt, is_valid):
        # breakpoint()

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


def pick_function(
    model: ModelOperator, prompt: str, functions: list[FunctionModel]
) -> FunctionModel | None:

    enhanced_prompt = EnhancedPrompt(
        "Pick one of these functions that satisfies the user's request:\n"
        + str(functions),
        json.dumps(prompt),
    )

    generated_name = "".join(
        generate_function_name(model, enhanced_prompt, functions)
    )

    function_chosen = {fn.name: fn for fn in functions}.get(generated_name)

    return function_chosen


def pick_parameters(
    model: ModelOperator, prompt: str, fn: FunctionModel
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}

    for parameter, type_model in fn.parameters.items():
        system = (
            f"The function to use is: {fn.model_dump_json()}.\n"
            f"Pick a value for the '{parameter}' parameter for the user's "
            + "request in JSON format."
        )

        assistant = (
            f'{json.dumps(parameters)[:-1]}, "{parameter}": '
            if parameters
            else f'{{"{parameter}": '
        )

        enhanced_prompt = EnhancedPrompt(
            system, json.dumps(prompt)[1:-1], assistant
        )

        print(f"{parameter}: ", end="", flush=True)

        generated_value = "".join(
            generate_parameter(model, enhanced_prompt, type_model.type)
        )

        parameters[parameter] = cast_value(generated_value, type_model.type)

    return parameters


def main() -> None:
    args_parser = build_parser()
    args = args_parser.parse_args()

    try:
        functions_definition = input_model_list(
            args.functions_definition, FunctionModel
        )
        prompts = input_model_list(args.input, PromptModel)
    except InputError as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

    model_operator = ModelOperator()
    function_calls: list[FunctionCall] = []

    for item in prompts:
        prompt = item.prompt

        print()
        print("—" * 42)
        print("With the prompt:")
        print(f"'{prompt}'")
        print()
        print("Asking to pick a function:")

        function_chosen = pick_function(
            model_operator, prompt, functions_definition
        )

        if not function_chosen:
            print("Error: Could not chose a function.", file=sys.stderr)
            sys.exit(1)

        print()
        print("-" * 24)
        print(
            "Asking for the parameters of the function",
            f"'{function_chosen.name}':",
        )

        parameters = pick_parameters(model_operator, prompt, function_chosen)

        print()
        print("The generated values are:", parameters)

        function_calls.append(
            FunctionCall(
                prompt=prompt, name=function_chosen.name, parameters=parameters
            )
        )

    try:
        output_model_list(args.output, function_calls)
    except OutputError as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
    else:
        print()
        print(f"Successfully dumped the function calls to '{args.output}'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
