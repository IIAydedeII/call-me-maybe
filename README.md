_This project has been created as part of the 42 curriculum by adede._

# call me maybe

> Introduction to function calling in LLMs

## Description

**call me maybe** is a Python project that picks and generates function calls based on a user-provided list of function definitions and prompts.

Small language models (SLMs) -like the 0.6B parameter model we use- are notoriously unreliable at generating structured output. When prompted to produce JSON, their success rates are considerably lower. To mitigate this behavior and to make sure a valid call is made, we use the method of **constrained decoding** — a technique that guides the model’s output token-by-token to guarantee valid structure, without relying on prompting alone.

The program performs the following steps:

1.  Imports the prompts and function definitions.
2.  Sets up the SLM via **llm_sdk** that is provided by the subject.
3.  Asks the model to pick a function and generate its parameters, complying to each prompt.
4.  Exports the resulting function calls.

## Instructions

### Makefile

The project includes a `Makefile` with the following commands:

| Command       | Description                                   |
| ------------- | --------------------------------------------- |
| `install`     | Install project dependencies                  |
| `run`         | Run the project                               |
| `debug`       | Run the project using `pdb`                   |
| `clean`       | Remove caches and temporary files             |
| `lint`        | Run basic `mypy` and `flake8` checks          |
| `lint-strict` | Run strict `mypy` and `flake8` checks         |
| `build`       | Builds the package to the root of the project |

The `uv` package manager is required for the preferred installation via the Makefile commands.

### Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install the project dependencies by running:

```bash
make install
```

### Running the Project

The main program is executed with:

```bash
make run
```

You can also use:

```bash
python -m src
```

You can specify the following arguments:

```bash
python -m src \
[--functions_definition <function_definition_file>] \
[--input <input_file>] \
[--output <output_file>]
```

## Prompt and Definition Files

The program processes two input files. By default, they are located in the `data/input/` directory.

- **Function Calling Tests**: contains a JSON array of natural language prompts that your system must process.
- **Functions Definition**: contains the available functions your system can call.

### File Structures

For **Function Calling Tests**, expected file structure is:

```json
[
  {
    "prompt": "<request>"
  }
  ...
]

```

For **Functions Definition**, expected file structure is:

```json
[
  {
    "name": "<function_name>",
    "description": "<description>",
    "parameters": {
      "<parameter>": {
        "type": "string|number|integer|boolean"
      }
      ...
    },
    "returns": {
      "type": "string|number|integer|boolean"
    }
  }
  ...
]
```

## Algorithm explanation

Language models generate text one token at a time. At each step, the model produces a _fitness_ distribution (_logits_) over all possible next tokens. Normally, you would sample from this distribution or pick the highest _fitness_ token. **Constrained decoding** intervenes in this process by modifying the _logits_ before token selection:

1. The model produces _logits_ for all possible tokens.
2. We identify which tokens would comply with the expected schema.
3. We set _logits_ for invalid tokens to negative infinity.
4. We get the most fit token only from the remaining valid tokens

## Design decisions

To maintain both a valid JSON structure and a performant generation procedure, we delay the JSON encoding to the end of the pipeline. We ask the SLM only to pick a function name from the available pool and to generate some values for the parameters of that said function's definition. With that, we make sure that the JSON structure is not affected by the outcome of the SLM's response, thus providing 100% JSON validity.

## Performance analysis

By asking only for the function name and the parameters, we skip asking for unnecessary tokens and we can keep the query prompt short. Function definitions are added to the query prompt only when asking the model to select a function, not when asking for prompts and vice versa.

We can then ensure that the JSON structure is reliable by modeling the data structures as Pydantic classes and using their JSON methods.

## Challenges faced

The hardest problem I faced was deciding where the prompt should end, or what should serve as the prompt terminator. Since we don't ask for a complete JSON response, we can't use the closing parentheses as an end of prompt specifier. I was hesitant to use double quotes either, since I couldn't figure how can I differentiate between a response ending quote and a quote that is part of the response string.

I was tempted to use _control tokens_ for a while and tested a whole variety of general purpose or special control tokens but it was more reliable to ask for parameters (especially string types) in JSON format in the end since the SLM is trained with it more regularly. It was easier to use _escaped double quotes_ (`\"`) as regular response quotes and any other quote as the prompt terminator.

## Testing strategy

I avoided modifying the prompt as much as possible because it has an unreliable influence over the resulting logit calculation.

I mainly used the example provided by the subject and made small variations to function names and descriptions. The parameter and return types weren't so much of a problem. I enjoyed creating similar function definitions and inspected how my prompting changes the outcome.

Besides import/export tests, I inspected the max logit token that I get and the logits of the tokens that I wanted the model to generate. Adding a `breakpoint()` right after the constrained decoding step was the most efficient way to inspect the tokens and their logits.

I wish I had automated the testing pipeline. It would also have been easier if I had implemented the demo functions myself.

## Example usage

We provide our request inputs as follows:

```json
[
  ...
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
  },
  ...
]
```

and present the available function definitions:

```json
[
  ...
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": {
        "type": "string"
      },
      "regex": {
        "type": "string"
      },
      "replacement": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  ...
]
```

You should expect a response like:

```txt
[ 9/11]
──────────────────────────────────────────
With the prompt:
'Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS'

Asking to pick a function:
'fn_substitute_string_with_regex'

Asking for the parameters:
source_string: 'Hello 34 I'm 233 years old'
regex: '([0-9]+)'
replacement: 'NUMBERS'

The generated values are: {"source_string": "Hello 34 I'm 233 years old", "regex": "([0-9]+)", "replacement": "NUMBERS"}
```

and an output file like so:

```json
[
  {
      "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
      "name": "fn_substitute_string_with_regex",
      "parameters": {
          "source_string": "Hello 34 I'm 233 years old",
          "regex": "([0-9]+)",
          "replacement": "NUMBERS"
      }
  },
  ...
]
```

## Bonuses

Since the bonuses are too vague, I can confidently confirm the implementation of only one of the bonuses mentioned in the subject:

- `[ ]` Support for multiple LLM models beyond Qwen/Qwen3-0.6B
- `[ ]` Recoding the tokenizer: avoiding direct use of encode and decode in the main code, instead using get_logits_from_input_ids and get_path_to_vocab_file
- `[ ]` Advanced error recovery mechanisms
- `[ ]` Performance optimizations (caching, batching)
- `[ ]` Comprehensive test suite
- `[X]` Visualization of the generation process
- `[ ]` Support for complex nested function arguments
- `[ ]` Public implementation of tokenizer encode and optional decode methods
- `[ ]` Demonstration of how encoding and decoding integrate with constrained decoding

## Resources

- [JSON Format](https://www.json.org/json-en.html)
- [Models | Pydantic](https://pydantic.dev/docs/validation/dev/concepts/)
- [A Guide to Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
- [Tensor Attributes | PyTorch](https://docs.pytorch.org/docs/2.14/tensor_attributes.html)
- [Control Tokens | LLM Knowledge Base](https://promptmetheus.com/resources/llm-knowledge-base/control-token)
- [Qwen3 Concepts (Special Control Tokens)](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md)
- [ChatML Format Guide | Ertas AI](https://www.ertas.ai/formats/chatml)

## AI Usage

AI was used to suggest small implementation ideas and explain concepts besides the online tools. Generated suggestions were reviewed, tested, modified, and integrated **manually**. No AI was used to generate the code **directly**; all functions are personally implemented.

---

42 Kocaeli
