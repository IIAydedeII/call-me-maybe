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


def parse_functions(file_path: str) -> list[FunctionModel]:
    try:
        with open(file_path, encoding="utf-8") as file:
            adapter = TypeAdapter(list[FunctionModel])
            functions = adapter.validate_json(file.read())
    except OSError as e:
        raise ParseError("Unable to open the specified file") from e
    except ValidationError as e:
        raise ParseError("Invalid data structure") from e

    return functions


class PromptModel(BaseModel):
    prompt: str


def parse_prompts(file_path: str) -> list[str]:
    try:
        with open(file_path, encoding="utf-8") as file:
            adapter = TypeAdapter(list[PromptModel])
            prompts = adapter.validate_json(file.read())
    except OSError as e:
        raise ParseError("Unable to open the specified file") from e
    except ValidationError as e:
        raise ParseError("Invalid data structure") from e

    return [item.prompt for item in prompts]


if __name__ == "__main__":
    print("Functions:")
    print(parse_functions("./data/input/functions_definition.json"))
    print()
    print("Prompts:")
    print(parse_prompts("./data/input/function_calling_tests.json"))
