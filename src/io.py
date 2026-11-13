from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, ValidationError, TypeAdapter
from .models import FunctionModel, PromptModel, AppError

T = TypeVar("T", bound=BaseModel)


def input_model_list(file_path: str, model: type[T]) -> list[T]:
    try:
        with open(file_path, encoding="utf-8") as file:
            adapter = TypeAdapter(list[model])
            models = adapter.validate_json(file.read())
    except OSError as e:
        raise AppError("Unable to open the specified file") from e
    except ValidationError as e:
        raise AppError("Invalid data structure") from e

    return models


def output_model_list(file_path: str, data: list[T]) -> None:
    path = Path(file_path)

    try:
        adapter = TypeAdapter(list[T])
        adapter.validate_python(data)
        json_data = adapter.dump_json(data, indent=4).decode("utf-8")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_data, encoding="utf-8")
    except OSError as e:
        raise AppError("Unable to write to the specified file") from e
    except ValidationError as e:
        raise AppError("Invalid data structure") from e


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
