import sys
import json
from collections.abc import Generator
from typing import Any
from llm_sdk import Small_LLM_Model
from .arguments import build_parser
from .input import input_model_list, InputError
from .models import FunctionModel, PromptModel, JSONType, FunctionCall
from .output import output_model_list, OutputError

MAX_ITERATION = 42
MODEL = Small_LLM_Model()


def generate_function_name(
    functions: list[FunctionModel], prompt: str
) -> Generator[str]:
    function_names = [fn.name for fn in functions]

    result = ""
    prompt_system = "\n".join(
        [
            "<|im_start|>system",
            f"Pick one of these functions that satisfies the user's request:",
            f"{str(functions)}",
            "<|im_end|>",
        ]
    )
    prompt_user = f"<|im_start|>user\n{json.dumps(prompt)}<|im_end|>"
    prompt_assistant = f"<|im_start|>assistant\n"
    enhanced_prompt = "\n".join([prompt_system, prompt_user, prompt_assistant])

    input_ids = MODEL.encode(enhanced_prompt).flatten().tolist()
    logits: list[float] = []
    for _ in range(MAX_ITERATION):
        logits = MODEL.get_logits_from_input_ids(input_ids)

        for token_id, logit in enumerate(logits):
            potential_token = MODEL.decode([token_id])

            contains = potential_token and any(
                name.startswith(result + potential_token)
                for name in function_names
            )

            if not contains:
                logits[token_id] = float("-inf")

        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        next_token = MODEL.decode([next_token_id])

        # print(f"Checking '{next_token}'")

        if not any(result + next_token in name for name in function_names):
            break

        yield next_token
        result += next_token
        input_ids.append(next_token_id)


def generate_parameter(prompt: str, json_type: JSONType) -> Generator[str]:
    result = ""

    input_ids = MODEL.encode(prompt).flatten().tolist()
    logits: list[float] = []
    quote_open = False
    for _ in range(MAX_ITERATION):
        logits = MODEL.get_logits_from_input_ids(input_ids)

        for token_id, logit in enumerate(logits):
            potential_token = MODEL.decode([token_id])

            # Allow/ignore all tokens with a double quote in them
            if '"' in potential_token:
                continue

            try:
                # Ensure the first token contains a double quote
                if not quote_open and not '"' in potential_token:
                    raise ValueError
                # Eliminate control tokens
                if not potential_token:
                    raise ValueError

                potential_result = result + potential_token
                match json_type:
                    case JSONType.NUMBER:
                        float(potential_result)
                    case JSONType.INTEGER:
                        int(potential_result)
                    case JSONType.BOOLEAN:
                        bool(potential_result)
                    case JSONType.STRING:
                        pass
            except ValueError:
                logits[token_id] = float("-inf")

        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
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
        prompt_system = "\n".join(
            [
                "<|im_start|>system",
                # "/no_think "
                f"The function to use is: {fn.model_dump_json()}.",
                f"Pick a value for the '{parameter}' parameter of this function "
                + "for the user's request in JSON format",
                "<|im_end|>",
            ]
        )
        prompt_user = f"<|im_start|>user\n{json.dumps(prompt)[1:-1]}<|im_end|>"
        prompt_assistant = "\n".join(
            [
                f"<|im_start|>assistant",
                (
                    f'{json.dumps(parameters)[:-1]}, "{parameter}": '
                    if parameters
                    else f'{{"{parameter}": '
                ),
            ]
        )
        enhanced_prompt = "\n".join(
            [prompt_system, prompt_user, prompt_assistant]
        )

        print(f"{parameter}: ", end="", flush=True)

        generated_value = ""
        print("'", end="", flush=True)
        for token in generate_parameter(enhanced_prompt, type_model.type):
            print(token, end="", flush=True)
            generated_value += token
        print("'")

        match type_model.type:
            case JSONType.NUMBER:
                generated_value = float(generated_value)
            case JSONType.INTEGER:
                generated_value = int(generated_value)
            case JSONType.BOOLEAN:
                generated_value = bool(generated_value)
            case JSONType.STRING:
                generated_value = json.loads(f'"{generated_value}"')

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
