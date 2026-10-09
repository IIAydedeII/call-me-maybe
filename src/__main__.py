from llm_sdk import Small_LLM_Model

ARBITRARY_GENERATION_LIMIT = 42

model = Small_LLM_Model()

prompt = "My name is"
response = ""
print("Prompt:", prompt)

token_ids = model.encode(prompt)
print()
print(token_ids)
print("-" * 10)
print("Shape:", token_ids.shape)
print("Dimentions:", token_ids.ndim)
print("Type:", token_ids.dtype)
print("Device:", token_ids.device)
print("Requires Gradient:", token_ids.requires_grad)

input_ids: list[int] = token_ids.flatten().tolist()

for _ in range(ARBITRARY_GENERATION_LIMIT):
    print()
    print("Input_IDs:", input_ids)

    logits = model.get_logits_from_input_ids(input_ids)
    print()
    print(
        f"Logits: [{logits[0]}, {logits[1]}, {logits[2]}, ..., {logits[-1]}]"
    )
    print("Length:", len(logits))

    next_token_id = max(range(len(logits)), key=lambda i: logits[i])

    decoded = model.decode([next_token_id])
    print()
    print(f"'{decoded}'")

    response += decoded

    input_ids.append(next_token_id)

print()
print(f"Response:\n{response}")
