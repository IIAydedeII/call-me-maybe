import sys
from collections.abc import Generator
from typing import Any
from llm_sdk import Small_LLM_Model, torch
from .arguments import build_parser
from .input import input_model_list, InputError
from .models import FunctionModel, PromptModel, JSONType, FunctionCall
from .output import output_model_list, OutputError
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


MAX_ITERATION = 42
MODEL = Small_LLM_Model()


def generate_function_name(
    functions: list[FunctionModel], prompt: str
) -> Generator[str]:
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

    input_ids = MODEL.encode(enhanced_prompt).flatten().tolist()
    for _ in range(MAX_ITERATION):
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

        yield next_token
        result += next_token
        input_ids.append(next_token_id)


def generate_parameter(prompt: str, json_type: JSONType) -> Generator[str]:
    result = ""

    input_ids = MODEL.encode(f"{prompt}").flatten().tolist()
    for _ in range(MAX_ITERATION):
        logits = MODEL.get_logits_from_input_ids(input_ids)
        STOP_TOKENS = {ControlToken.IM_END, ControlToken.EOS}
        IGNORE_TOKENS = {ControlToken.think, ControlToken.think_end}

        for token_id, logit in enumerate(logits):
            # if token_id in STOP_TOKENS:
            #     print("END:", logit)
            potential_token = control_decode(MODEL, token_id)

            if result and token_id in STOP_TOKENS:
                continue

            if token_id in IGNORE_TOKENS:
                continue

            # logits[151643] = float("-inf")
            # logits[151667] = float("-inf")
            # logits[151666] = float("-inf")
            # logits[151668] = float("-inf")

            # if not potential_token:
            #     logits[token_id] = float("-inf")
            # if potential_token.endswith('"'):
            #     continue

            try:
                match json_type:
                    case JSONType.NUMBER:
                        # if potential_token.strip() != potential_token:
                        #     raise ValueError
                        float(potential_token)
                    case JSONType.INTEGER:
                        # if potential_token.strip() != potential_token:
                        #     raise ValueError
                        int(potential_token)
                    case JSONType.BOOLEAN:
                        # if potential_token.strip() != potential_token:
                        #     raise ValueError
                        bool(potential_token)
                    case JSONType.STRING:
                        pass
            except ValueError:
                logits[token_id] = float("-inf")

        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        next_token = control_decode(MODEL, next_token_id)
        # print(f"\n'{next_token_id:6}':", logits[next_token_id])
        # for id in STOP_TOKENS:
        #     print(f"'{id}':", logits[id])

        # print(f"Checking '{next_token}'")
        # breakpoint()

        if next_token_id in STOP_TOKENS:
            break
        # if '"' in next_token:
        #     next_token = next_token[: next_token.find('"')]
        #     yield next_token
        #     break

        result += next_token
        input_ids.append(next_token_id)
        if not any(
            [
                next_token_id in IGNORE_TOKENS,
                # not next_token.strip(),
            ]
        ):
            yield next_token


def generate_parameters(fn: FunctionModel, prompt: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {}

    for parameter, type_model in fn.parameters.items():
        prompt_system = "\n".join(
            [
                "<|im_start|>system",
                "/no_think "
                f"The function to use is: {fn.model_dump_json()}.",
                f"Current values are: {parameters}" if parameters else "",
                # f"End your answer with '<|endoftext|>'",
                f"Pick a value for the '{parameter}' parameter of this function for the user's request.",
                # "Wrap the value you picked in double quotes.",
                "<|im_end|>",
            ]
        )
        prompt_user = f"<|im_start|>user\n{prompt}<|im_end|>"
        enhanced_prompt = "\n".join(
            [
                prompt_system,
                prompt_user,
                f'<|im_start|>tool\n<think>\n\n</think>\n{{"{parameter}":',
                # f"<|im_start|>tool\n",
            ]
        )

        print("_" * 24)
        print(f"'{parameter}': ", end="", flush=True)

        generated_value = ""
        print("'", end="", flush=True)
        # if "with_regex" in fn.name:
        #     print("\nPROMPT:", enhanced_prompt)
        for token in generate_parameter(enhanced_prompt, type_model.type):
            print(token, end="", flush=True)
            generated_value += token
        print("'")

        # match type_model.type:
        #     case JSONType.NUMBER:
        #         generated_value = float(generated_value)
        #     case JSONType.INTEGER:
        #         generated_value = int(generated_value)
        #     case JSONType.BOOLEAN:
        #         generated_value = bool(generated_value)
        #     case JSONType.STRING:
        #         pass
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
        print("-" * 42)
        print("With the prompt:")
        print(f"'{prompt}'")
        # print()
        # print("Asking for a function name:")

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

        # print()
        # print("The chosen function is:", repr(function_chosen))

        if not function_chosen:
            print("Error: Could not chose a function.", file=sys.stderr)
            sys.exit(1)

        # print()
        # print(
        #     f"Asking for the parameters of the function '{function_chosen.name}':"
        # )

        parameters = generate_parameters(function_chosen, prompt)

        # print()
        # print("The generated values are:", parameters)

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
