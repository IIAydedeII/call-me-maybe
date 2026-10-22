from pathlib import Path
from typing import TypeVar
from pydantic import BaseModel, ValidationError, TypeAdapter


class OutputError(Exception):
    """Raised when the data cannot be written to the file in JSON format."""


T = TypeVar("T", bound=BaseModel)


def output_model_list(file_path: str, data: list[T]) -> None:
    path = Path(file_path)

    try:
        adapter = TypeAdapter(list[T])
        adapter.validate_python(data)
        json_data = adapter.dump_json(data, indent=4).decode("utf-8")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_data, encoding="utf-8")
    except OSError as e:
        raise OutputError("Unable to write to the specified file") from e
    except ValidationError as e:
        raise OutputError("Invalid data structure") from e
