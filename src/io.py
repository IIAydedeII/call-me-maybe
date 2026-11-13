"""Read and write lists of Pydantic models using JSON files."""

from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, ValidationError, TypeAdapter
from .models import FunctionModel, PromptModel
from .utils import AppError

T = TypeVar("T", bound=BaseModel)


def input_model_list(file_path: str, model: type[T]) -> list[T]:
    """Load and validate a list of models from a JSON file.

    Args:
        file_path (str): Path to the JSON file to read.
        model (type[T]):
            Pydantic model class used to validate each list element.

    Returns:
        list[T]: A list of validated instances of the specified model.

    Raises:
        AppError:
            If the file cannot be opened or read or if the JSON data does not
            match the expected model structure.
    """
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
    """Validate and write a list of models to a JSON file.

    Creates any missing parent directories before writing the file.

    Args:
        file_path (str): Path to the JSON file to write.
        data (list[T]): List of Pydantic model instances to serialize.

    Raises:
        AppError:
            If the file or its parent directories cannot be accessed or if the
            provided data does not conform to its expected Pydantic model
            structure.
    """

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
