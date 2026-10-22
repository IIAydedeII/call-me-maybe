from typing import TypeVar
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
    parameters: dict[str, TypeModel]
    returns: TypeModel


class PromptModel(BaseModel):
    prompt: str


T = TypeVar("T", bound=BaseModel)


def parse_model_list(file_path: str, model: type[T]) -> list[T]:
    try:
        with open(file_path, encoding="utf-8") as file:
            adapter = TypeAdapter(list[model])
            models = adapter.validate_json(file.read())
    except OSError as e:
        raise ParseError("Unable to open the specified file") from e
    except ValidationError as e:
        raise ParseError("Invalid data structure") from e

    return models


if __name__ == "__main__":
    print("Functions:")
    print(
        parse_model_list(
            "./data/input/functions_definition.json", FunctionModel
        )
    )
    print()
    print("Prompts:")
    print(
        parse_model_list(
            "./data/input/function_calling_tests.json", PromptModel
        )
    )
