import sys
import json
from collections.abc import Generator
from typing import Any
from llm_sdk import Small_LLM_Model
from .arguments import build_parser
from .input import input_model_list, InputError
from .models import FunctionModel, PromptModel, JSONType, FunctionCall
from .utils import build_prompt
from .output import output_model_list, OutputError

MAX_ITERATION = 42
MODEL = Small_LLM_Model()


def generate_function_name(
    functions: list[FunctionModel], prompt: str
) -> Generator[str]:
    function_names = [fn.name for fn in functions]

    enhanced_prompt = build_prompt(
        "Pick one of these functions that satisfies the user's request:\n"
        + str(functions),
        json.dumps(prompt),
    )

    result = ""
    input_ids = MODEL.encode(enhanced_prompt).flatten().tolist()
    for _ in range(MAX_ITERATION):
        logits = MODEL.get_logits_from_input_ids(input_ids)

        for token_id, logit in enumerate(logits):
            token = MODEL.decode([token_id])

            contains = token and any(
                name.startswith(result + token) for name in function_names
            )

            if not contains:
                logits[token_id] = float("-inf")

        next_token_id = logits.index(max(logits))
        next_token = MODEL.decode([next_token_id])

        if not any(
            name.startswith(result + next_token) for name in function_names
        ):
            break

        yield next_token
        result += next_token
        input_ids.append(next_token_id)


def generate_parameter(prompt: str, json_type: JSONType) -> Generator[str]:
    quote_open = False

    result = ""
    input_ids = MODEL.encode(prompt).flatten().tolist()
    for _ in range(MAX_ITERATION):
        logits = MODEL.get_logits_from_input_ids(input_ids)

        for token_id, logit in enumerate(logits):
            token = MODEL.decode([token_id])

            # Allow/ignore all tokens with a double quote in them
            if '"' in token:
                continue

            try:
                # Ensure the first token contains a double quote
                if not quote_open and not '"' in token:
                    raise ValueError
                # Eliminate control tokens
                if not token:
                    raise ValueError

                potential_result = result + token
                cast_value(potential_result, json_type)
            except (ValueError, json.decoder.JSONDecodeError):
                logits[token_id] = float("-inf")

        next_token_id = logits.index(max(logits))
        next_token = MODEL.decode([next_token_id])

        # breakpoint()

        if not quote_open and '"' in next_token:
            quote_open = True

            # Remove the part before the opening quote
            quote_index = next_token.find('"')
            next_token = next_token[quote_index + 1 :]

        if quote_open and '"' in next_token and not '\\"' in next_token:
            # Remove the part after the closing quote
            yield next_token[: next_token.find('"')]
            break

        yield next_token
        result += next_token
        input_ids.append(next_token_id)


def generate_parameters(fn: FunctionModel, prompt: str) -> dict[str, Any]:
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
        enhanced_prompt = build_prompt(
            system, json.dumps(prompt)[1:-1], assistant
        )

        print(f"{parameter}: ", end="", flush=True)

        generated_value = ""
        print("'", end="", flush=True)
        for token in generate_parameter(enhanced_prompt, type_model.type):
            print(token, end="", flush=True)
            generated_value += token
        print("'")

        generated_value = cast_value(generated_value, type_model.type)

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
        print("—" * 42)
        print("With the prompt:")
        print(f"'{prompt}'")
        print()
        print("Asking to pick a function:")

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

        if not function_chosen:
            print("Error: Could not chose a function.", file=sys.stderr)
            sys.exit(1)

        print()
        print("-" * 24)
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
