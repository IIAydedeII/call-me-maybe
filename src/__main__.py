import sys
import json
import time
from datetime import timedelta
from typing import Any
from .arguments import build_parser
from .io import input_model_list, output_model_list
from .models import FunctionModel, PromptModel, FunctionCall, EnhancedPrompt
from .generators import generate_function_name, generate_parameter
from .utils import cast_value, ModelOperator, print_generator, AppError

generate_function_name = print_generator(generate_function_name)
generate_parameter = print_generator(generate_parameter)


def pick_function(
    model: ModelOperator, prompt: str, functions: list[FunctionModel]
) -> FunctionModel | None:
    enhanced_prompt = EnhancedPrompt(
        "Pick one of these functions that satisfies the user's request:\n"
        + str(functions),
        json.dumps(prompt)[1:-1],
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

    print("[ IMPORTING ]")
    print("═" * 42)

    functions = input_model_list(args.functions_definition, FunctionModel)
    print(len(functions), "function definitions imported")

    prompts = input_model_list(args.input, PromptModel)
    print(len(prompts), "prompts collected")

    print()
    print()
    print("[ INITIALIZING THE MODEL ]")
    print("═" * 42)
    model_operator = ModelOperator()

    print()
    print()
    print("[ PROCESSING THE PROMPTS ]")
    print("═" * 42)
    start = time.perf_counter()

    function_calls: list[FunctionCall] = []

    for index, item in enumerate(prompts):
        prompt = item.prompt

        print(f"[{index+1:-2}/{len(prompts):-2}]")
        print("─" * 42)
        print("With the prompt:")
        print(f"'{prompt}'")
        print()
        print("Asking to pick a function:")
        chosen_function = pick_function(model_operator, prompt, functions)

        if not chosen_function:
            raise AppError("Could not choose a function.")

        print()
        print("Asking for the parameters:")
        parameters = pick_parameters(model_operator, prompt, chosen_function)

        print()
        print("The generated values are:", json.dumps(parameters))
        print()

        function_calls.append(
            FunctionCall(
                prompt=prompt, name=chosen_function.name, parameters=parameters
            )
        )

    elapsed = time.perf_counter() - start
    print("═" * 42)
    print("Processed the prompts in:", timedelta(seconds=elapsed))

    print()
    print()
    print("[ OUTPUTTING ]")
    print("═" * 42)
    output_model_list(args.output, function_calls)

    print(f"Successfully dumped the function calls to '{args.output}'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except AppError as e:
        print("[ERROR]:", e, file=sys.stderr)
        sys.exit(1)
