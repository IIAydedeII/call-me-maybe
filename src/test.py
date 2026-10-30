from llm_sdk import Small_LLM_Model
from enum import IntEnum, StrEnum, auto


class ControlToken(IntEnum):
    EOS = 151643
    # EOS = 0
    IM_START = auto()
    IM_END = auto()
    object_ref_start = auto()
    object_ref_end = auto()
    box_start = auto()
    box_end = auto()
    quad_start = auto()
    quad_end = auto()
    vision_start = auto()
    vision_end = auto()
    vision_pad = auto()
    image_pad = auto()
    video_pad = auto()
    tool_call = auto()
    tool_call_end = auto()
    fim_prefix = auto()
    fim_middle = auto()
    fim_suffix = auto()
    fim_pad = auto()
    repo_name = auto()
    file_sep = auto()
    tool_response = auto()
    tool_response_end = auto()
    think = auto()
    think_end = auto()


class ModelName(StrEnum):
    QWEN_600 = "Qwen/Qwen3-0.6B"
    QWEN_1700 = "Qwen/Qwen3-1.7B"
    QWEN_WEB = "Qwen/WebWorld-8B"
    HF_135 = "HuggingFaceTB/SmolLM2-135M"
    HF_360 = "HuggingFaceTB/SmolLM2-360M-Instruct"


def control_decode(model: Small_LLM_Model, id: int) -> str:
    match id:
        case ControlToken.EOS:
            return "<|endoftext|>"
        case ControlToken.IM_START:
            return "<|im_start|>"
        case ControlToken.IM_END:
            return "<|im_end|>"
        case ControlToken.object_ref_start:
            return "<|object_ref_start|>"
        case ControlToken.object_ref_end:
            return "<|object_ref_end|>"
        case ControlToken.box_start:
            return "<|box_start|>"
        case ControlToken.box_end:
            return "<|box_end|>"
        case ControlToken.quad_start:
            return "<|quad_start|>"
        case ControlToken.quad_end:
            return "<|quad_end|>"
        case ControlToken.vision_start:
            return "<|vision_start|>"
        case ControlToken.vision_end:
            return "<|vision_end|>"
        case ControlToken.vision_pad:
            return "<|vision_pad|>"
        case ControlToken.image_pad:
            return "<|image_pad|>"
        case ControlToken.video_pad:
            return "<|video_pad|>"
        case ControlToken.tool_call:
            return "<tool_call>"
        case ControlToken.tool_call_end:
            return "</tool_call>"
        case ControlToken.fim_prefix:
            return "<|fim_prefix|>"
        case ControlToken.fim_middle:
            return "<|fim_middle|>"
        case ControlToken.fim_suffix:
            return "<|fim_suffix|>"
        case ControlToken.fim_pad:
            return "<|fim_pad|>"
        case ControlToken.repo_name:
            return "<|repo_name|>"
        case ControlToken.file_sep:
            return "<|file_sep|>"
        case ControlToken.tool_response:
            return "<tool_response>"
        case ControlToken.tool_response_end:
            return "</tool_response>"
        case ControlToken.think:
            return "<think>"
        case ControlToken.think_end:
            return "</think>"
        case _:
            return model.decode([id])


few_shot = """
<|im_start|>system
You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09-01
Current date: 2023-03-01<|im_end|>
<|im_start|>user
How are you<|im_end|>
<|im_start|>assistant
I am doing well!<|im_end|>
<|im_start|>user
How are you now?<|im_end|>
"""

tool_calling_system = """<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> [{"type": "function", "function": {"name": "get_stock_fundamentals", "description": "Get fundamental data for a given stock symbol using yfinance API.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}] </tools> Use the following pydantic model json schema for each tool call you will make: {"title": "FunctionCall", "type": "object", "properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, "required": ["name", "arguments"]} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</tool_call><|im_end|>"""

tool_calling_system = """<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> [{"type": "function", "function": {"name": "fn_multiply_numbers", "description": "Multiply two numbers together and return their product.", "parameters": {"a": {"type": "number"}, "b": {"type": "number"}}}}] </tools> Use the following pydantic model json schema for each tool call you will make: {"title": "FunctionCall", "type": "object", "properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, "required": ["name", "arguments"]} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": fn_stock_data, "arguments": <args-dict>}
</tool_call><|im_end|>"""

tool_calling_user = """<|im_start|>user
What is the product of 12 and 4?<|im_end|>"""

tool_calling_system = """"<|im_start|>system
The function to use is: 'FunctionModel(name='fn_multiply_numbers', description='Multiply two numbers together and return their product.', parameters={'a': {type: 'number'}, 'b': {type: 'number'}}, returns={type: 'number'})'
Current values: {'a': None, 'b': None}
End your answer with '<|endoftext|>' or '<|im_end|>'.
Pick a value for the 'a' parameter of this function for the given request<|im_end|>"""

tool_calling_user = """<|im_start|>user
What is the product of 3 and 5?<|im_end|>"""

tool_calling_system = """"<|im_start|>system
The function to use is: '{"name": "fn_format_template", "description": "Format a template string containing placeholders.", "parameters": {"template": {"type": "string"}}, "returns": {"type": "string"}}'.
Pick a value for the 'template' parameter of this function for the user's request.
Wrap the value you picked in double quotes.
<|im_end|>"""

tool_calling_user = """<|im_start|>user
Format template: Say \"hello\" to {name}<|im_end|>"""

tool_calling_system = """"<|im_start|>system
The function to use is: '{"name": "fn_format_template", "description": "Format a template string containing placeholders.", "parameters": {"template": {"type": "string"}}, "returns": {"type": "string"}}'.
Pick a value for the 'template' parameter of this function for the user's request.
Wrap the value you picked in double quotes.
<|im_end|>"""

tool_calling_user = """<|im_start|>user
Format template: Say \"hello\" to {name}<|im_end|>"""


def main():
    model = Small_LLM_Model(model_name=ModelName.QWEN_600)
    # print(model.get_path_to_tokenizer_file())
    # print(model.get_path_to_vocab_file())
    # print(model.get_path_to_merges_file())

    # for text in [
    #     "<|im_start|>",
    #     "<|im_sep|>",
    #     "<|im_end|>",
    #     "\n<|im_start|>assistant\n",
    #     "\n<|im_start|>assistant<|im_sep|>",
    # ]:
    #     ids = model.encode(text)
    #     print(repr(text), ids)

    # for text in ["20", "２０", "3", "３"]:
    #     ids = model.encode(text)
    #     print(repr(text), [hex(ord(c)) for c in text], ids)

    prompt = "\n".join(
        [
            tool_calling_system,
            tool_calling_user,
            "<|im_start|>assistant\n<think>\n\n</think>\n",
        ]
    )

    print(prompt)
    print("_" * 42)
    while True:
        input_ids = model.encode(prompt).flatten().tolist()
        logits = model.get_logits_from_input_ids(input_ids)
        logits[151667] = float("-inf")
        logits[ControlToken.IM_START] = float("-inf")
        next_token_id = logits.index(max(logits))
        if next_token_id in (ControlToken.EOS, ControlToken.IM_END):
            break

        next_token = control_decode(model, next_token_id)
        if not next_token:
            print("ZORT:", next_token_id)

        print(next_token, end="", flush=True)
        prompt += next_token


try:
    main()
except KeyboardInterrupt:
    pass
