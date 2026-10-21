import json
from typing import Any
from enum import StrEnum, auto
from pydantic import BaseModel, Field, ValidationError, TypeAdapter


class JSONType(StrEnum):
    STRING = auto()
    BOOLEAN = auto()
    INTEGER = auto()
    NUMBER = auto()


class ParseError(Exception):
    """Raised when a file cannot be read or its contents cannot be parsed."""


class TypeModel(BaseModel):
    json_type: JSONType = Field(alias="type")

    def __repr__(self) -> str:
        return f"{{type: '{self.json_type}'}}"


class FunctionModel(BaseModel):
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, TypeModel] = Field(min_length=1)
    returns: TypeModel


def parse_functions(file_path: str) -> list[FunctionModel]:
    functions: list[FunctionModel] = []

    try:
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
    except OSError as e:
        raise ParseError("Unable to open the specified file") from e
    except json.decoder.JSONDecodeError as e:
        raise ParseError("Could not parse the file as JSON") from e

    try:
        type_adapter = TypeAdapter(list[dict[str, Any]])
        type_adapter.validate_python(data)

        for fn in data:
            functions.append(FunctionModel(**fn))
    except ValidationError as e:
        raise ParseError("Invalid data structure") from e

    return functions


if __name__ == "__main__":
    print(parse_functions("./data/input/functions_definition.json"))
