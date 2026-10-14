from llm_sdk import Small_LLM_Model, torch
from collections import deque


def iterate_prompt() -> None:
    PROMPT = "Slim Shady is one of the most influential figures that exist in fiction."
    ITERATION = 420

    model = Small_LLM_Model()
    # response = ""
    token_ids_tensor = model.encode(PROMPT)
    token_ids: list[int] = token_ids_tensor.flatten().tolist()
    queue_length = len(token_ids) * 2

    print()
    print("Iteration Count:", ITERATION)
    print("Queue Length:", queue_length)
    print("-" * 10)
    print("Type:", token_ids_tensor.dtype)
    print("Device:", token_ids_tensor.device)
    print("-" * 10)
    print(PROMPT)
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
    iterate_prompt()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
