def build_prompt(system: str, user: str, assistant: str = "") -> str:
    return "\n".join(
        [
            "<|im_start|>system",
            f"{system}<|im_end|>",
            "<|im_start|>user",
            f"{user}<|im_end|>",
            "<|im_start|>assistant",
            f"{assistant}",
        ]
    )
