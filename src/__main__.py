import sys
from collections.abc import Generator
from typing import Any
from llm_sdk import Small_LLM_Model, torch
from .arguments import build_parser
from .input import input_model_list, InputError
from .models import FunctionModel, PromptModel, JSONType, FunctionCall
from .output import output_model_list, OutputError

MAX_ITERATION = 42
MODEL = Small_LLM_Model()


def generate_function_name(
    functions: list[FunctionModel], prompt: str
) -> Generator[str]:
    FUNCTION_NAMES = [fn.name for fn in functions]

    result = ""
    enhanced_prompt = "\n".join(
        [
            str(functions),
            "",
            f"request: {prompt}",
            "",
            "One of these functions' name above "
            + "to satisfy the given request is ",
        ]
    )

    token_ids_tensor = MODEL.encode(enhanced_prompt)
    token_ids: list[int] = token_ids_tensor.flatten().tolist()
    input_ids = token_ids
    for _ in range(MAX_ITERATION):
        logits = MODEL.get_logits_from_input_ids(input_ids)

        for token_id, logit in enumerate(logits):
            potential_token = MODEL.decode([token_id])

            contains = potential_token and any(
                name.startswith(result + potential_token)
                for name in FUNCTION_NAMES
            )

            if not contains:
                logits[token_id] = float("-inf")

        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        next_token = MODEL.decode([next_token_id])

        # print(f"Checking '{next_token}'")

        if not any(result + next_token in name for name in FUNCTION_NAMES):
            break

        yield next_token
        result += next_token
        input_ids.append(next_token_id)


def generate_parameter(prompt: str, json_type: JSONType) -> Generator[str]:
    result = ""

    input_ids = MODEL.encode(prompt + '"').flatten().tolist()
    for _ in range(MAX_ITERATION):
        logits = MODEL.get_logits_from_input_ids(input_ids)

        for token_id, logit in enumerate(logits):
            potential_token = MODEL.decode([token_id])

            if potential_token.endswith('"'):
                break

            try:
                match json_type:
                    case JSONType.NUMBER:
                        float(result + potential_token)
                    case JSONType.INTEGER:
                        int(result + potential_token)
                    case JSONType.BOOLEAN:
                        bool(result + potential_token)
                    case JSONType.STRING:
                        if '"' in potential_token:
                            raise ValueError
            except ValueError:
                logits[token_id] = float("-inf")

        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        next_token = MODEL.decode([next_token_id])

        # print(f"Checking '{next_token}'")

        if '"' in next_token:
            break

        yield next_token
        result += next_token
        input_ids.append(next_token_id)


def generate_parameters(fn: FunctionModel, prompt: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {param: None for param in fn.parameters}

    for parameter, type_model in fn.parameters.items():
        enhanced_prompt = "\n".join(
            [
                repr(fn),
                "",
                f"\nCurrent values: {parameters}\n" if parameters else "",
                f"Request: {prompt}",
                f"The value for the '{parameter}' parameter of this function"
                + " for the given request is ",
            ]
        )
        print(f"'{parameter}': ", end="", flush=True)

        generated_value = ""
        print("'", end="", flush=True)
        # if "with_regex" in fn.name:
        #     print("\nPROMPT:", enhanced_prompt)
        for token in generate_parameter(enhanced_prompt, type_model.json_type):
            print(token, end="", flush=True)
            generated_value += token
        print("'")
        parameters.update({parameter: generated_value})

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

    function_calls: list[FunctionCall] = []

    for item in prompts:
        prompt = item.prompt

        print()
        print("-" * 42)
        print("With the prompt:")
        print(f"'{prompt}'")
        print()
        print("Asking for a function name:")

        generated_name = ""
        print("'", end="", flush=True)
        for token in generate_function_name(functions_definition, prompt):
            print(token, end="", flush=True)
            generated_name += token
        print("'")

        function_chosen = next(
            (fn for fn in functions_definition if fn.name == generated_name),
            None,
        )

        print()
        print("The chosen function is:", repr(function_chosen))

        if not function_chosen:
            print("Error: Could not chose a function.", file=sys.stderr)
            sys.exit(1)

        print()
        print(
            f"Asking for the parameters of the function '{function_chosen.name}':"
        )

        parameters = generate_parameters(function_chosen, prompt)

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
