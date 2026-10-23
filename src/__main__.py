import sys
from llm_sdk import Small_LLM_Model, torch
from .arguments import build_parser
from .input import input_model_list, InputError
from .models import FunctionModel, PromptModel, JSONType, FunctionCall
from .output import output_model_list, OutputError

MODEL = Small_LLM_Model()


def choose_function_from_list(
    functions: list[FunctionModel],
    prompt: str,
) -> FunctionModel | None:
    ITERATION = 42
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
    for _ in range(ITERATION):
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

        result += next_token
        input_ids.append(next_token_id)

    # print()
    # print(f"Looking for '{result}' in the function definitions")

    return next((fn for fn in functions if fn.name == result), None)


def generate_parameters(fn: FunctionModel, prompt: str) -> dict[str, str]:
    parameters: dict[str, str] = {key: "" for key in fn.parameters}

    for parameter, type_model in fn.parameters.items():
        result = ""
        enhanced_prompt = "\n".join(
            [
                repr(fn),
                "",
                f"request: {prompt}",
                (
                    ""
                    if all([not v for v in parameters.values()])
                    else f"\nChosen values: {parameters}"
                ),
                "",
                f"The value for the '{parameter}' parameter of this function for the given request is ",
            ]
        )

        print(f"Asking for the parameter '{parameter}'")

        ITERATION = 42

        token_ids_tensor = MODEL.encode(enhanced_prompt + '"')
        token_ids: list[int] = token_ids_tensor.flatten().tolist()
        input_ids = token_ids
        for _ in range(ITERATION):
            logits = MODEL.get_logits_from_input_ids(input_ids)

            for token_id, logit in enumerate(logits):
                potential_token = MODEL.decode([token_id])

                if potential_token.endswith('"'):
                    break

                try:
                    match type_model.json_type:
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

            result += next_token
            input_ids.append(next_token_id)

        parameters[parameter] = result

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
        print("Asking for a function name for the prompt:")
        print(f"'{prompt}'")

        function_chosen = choose_function_from_list(
            functions_definition, prompt
        )

        print()
        print("The chosen function is:", repr(function_chosen))

        if not function_chosen:
            print("Error: Could not chose a function.", file=sys.stderr)
            sys.exit(1)

        print()
        print(
            f"Asking for the parameters of the function '{function_chosen.name}'"
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
