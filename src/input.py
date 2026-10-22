from typing import TypeVar
from pydantic import BaseModel, ValidationError, TypeAdapter
from .models import FunctionModel, PromptModel


class InputError(Exception):
    """Raised when a file cannot be read or its contents cannot be parsed."""


T = TypeVar("T", bound=BaseModel)


def input_model_list(file_path: str, model: type[T]) -> list[T]:
    try:
        with open(file_path, encoding="utf-8") as file:
            adapter = TypeAdapter(list[model])
            models = adapter.validate_json(file.read())
    except OSError as e:
        raise InputError("Unable to open the specified file") from e
    except ValidationError as e:
        raise InputError("Invalid data structure") from e

    return models


if __name__ == "__main__":
    print("Functions:")
    print(
        input_model_list(
            "./data/input/functions_definition.json", FunctionModel
        )
    )
    print()
    print("Prompts:")
    print(
        input_model_list(
            "./data/input/function_calling_tests.json", PromptModel
        )
    )
