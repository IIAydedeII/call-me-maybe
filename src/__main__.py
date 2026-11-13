import sys
import json
from typing import Any
from .arguments import build_parser
from .input import input_model_list, InputError
from .models import FunctionModel, PromptModel, FunctionCall, EnhancedPrompt
from .generators import generate_function_name, generate_parameter
from .utils import cast_value, ModelOperator, print_generator
from .output import output_model_list, OutputError

generate_function_name = print_generator(generate_function_name)
generate_parameter = print_generator(generate_parameter)

def pick_function(
    model: ModelOperator, prompt: str, functions: list[FunctionModel]
) -> FunctionModel | None:
    enhanced_prompt = EnhancedPrompt(
        "Pick one of these functions that satisfies the user's request:\n"
        + str(functions),
        json.dumps(prompt),
    )

    name = "".join(generate_function_name(model, enhanced_prompt, functions))

    return {fn.name: fn for fn in functions}.get(name)


def pick_parameters(
    model: ModelOperator, prompt: str, fn: FunctionModel
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}

    user = json.dumps(prompt)[1:-1]

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

        enhanced_prompt = EnhancedPrompt(system, user, assistant)

        print(f"{parameter}: ", end="", flush=True)

        value = "".join(
            generate_parameter(model, enhanced_prompt, type_model.type)
        )

        parameters[parameter] = cast_value(value, type_model.type)

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
