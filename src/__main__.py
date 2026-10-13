from llm_sdk import Small_LLM_Model

ARBITRARY_GENERATION_LIMIT = 42

model = Small_LLM_Model()

prompt = "My name is"
# response = ""

token_ids = model.encode(prompt)
print()
print(token_ids)
print("-" * 10)
print("Shape:", token_ids.shape)
print("Dimentions:", token_ids.ndim)
print("Type:", token_ids.dtype)
print("Device:", token_ids.device)
print("Requires Gradient:", token_ids.requires_grad)
print()

input_ids: list[int] = token_ids.flatten().tolist()

print(prompt, end="")

for _ in range(ARBITRARY_GENERATION_LIMIT):
    logits = model.get_logits_from_input_ids(input_ids)
    next_token_id = max(range(len(logits)), key=lambda i: logits[i])
    decoded = model.decode([next_token_id])
    print(decoded, end="", flush=True)

    # response += decoded
    input_ids.append(next_token_id)

# print()
# print(f"Response:\n{response}")
