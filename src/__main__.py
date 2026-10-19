from llm_sdk import Small_LLM_Model, torch
from collections import deque
from .functions import parse_functions, ParseError


def iterate_prompt(prompt: str) -> None:
    enhanced_prompt = (
        prompt
        + "\nPick one of these functions above"
        + " that roots the square and show in JSON format:"
    )
    ITERATION = 420

    model = Small_LLM_Model()
    # response = ""
    token_ids_tensor = model.encode(enhanced_prompt)
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
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        decoded = model.decode([next_token_id])
        print(decoded, end="", flush=True)

        # response += decoded
        input_ids.append(next_token_id)

    # print()
    # print(f"Response:\n{response}")


def main() -> None:
    try:
        functions_definition = parse_functions(
            "./data/input/functions_definition.json"
        )
        iterate_prompt(str(functions_definition))
    except ParseError as e:
        print("Error:", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
