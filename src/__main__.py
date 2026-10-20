from llm_sdk import Small_LLM_Model, torch
from collections import deque
from .functions import parse_functions, ParseError, FunctionModel

PROMPT = 'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS'
MODEL = Small_LLM_Model()


def iterate_prompt(prompt: str) -> None:
    enhanced_prompt = (
        prompt
        + "\nPick one of these functions above"
        + " that roots the square and show in JSON format:"
    )
    ITERATION = 420

    # response = ""
    token_ids_tensor = MODEL.encode(enhanced_prompt)
    token_ids: list[int] = token_ids_tensor.flatten().tolist()
    queue_length = len(token_ids) * 2

    print()
    print("Iteration Count:", ITERATION)
    print("Queue Length:", queue_length)
    print("-" * 10)
    print("Type:", token_ids_tensor.dtype)
    print("Device:", token_ids_tensor.device)
    print("-" * 10)
    print(enhanced_prompt)
    print()

    input_ids = deque(token_ids, maxlen=queue_length)

    for _ in range(ITERATION):
        logits = MODEL.get_logits_from_input_ids(input_ids)
        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        decoded = MODEL.decode([next_token_id])
        print(decoded, end="", flush=True)

        # response += decoded
        input_ids.append(next_token_id)

    # print()
    # print(f"Response:\n{response}")


def choose_function_from_list(
    functions: list[FunctionModel],
) -> FunctionModel | None:
    ITERATION = 10
    FUNCTION_NAMES = [fn.name for fn in functions]

    result = ""
    enhanced_prompt = "\n".join(
        [
            str(functions),
            "",
            f"request: {PROMPT}",
            "",
            "One of these functions' name above "
            + "to satisfy the given request is ",
        ]
    )

    print()
    print(f"Prompt:\n'{enhanced_prompt}'")
    print()
    print("Asking for the function name")

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

        print(f"Checking '{next_token}'")

        if not any(result + next_token in name for name in FUNCTION_NAMES):
            break

        result += next_token
        input_ids.append(next_token_id)

    print()
    print(f"Looking for '{result}'")

    return next((fn for fn in functions if fn.name == result), None)


def main() -> None:
    try:
        functions_definition = parse_functions(
            "./data/input/functions_definition.json"
        )
        # iterate_prompt(str(functions_definition))

        function_chosen = choose_function_from_list(functions_definition)

        print()
        print("The chosen function is:", repr(function_chosen))
    except ParseError as e:
        print("Error:", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
