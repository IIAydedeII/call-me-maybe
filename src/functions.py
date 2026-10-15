import json
from typing import Literal, Any
from pydantic import BaseModel, Field, ValidationError, TypeAdapter

Type = Literal["string", "number"]


class TypeModel(BaseModel):
    type_: Type = Field(alias="type")


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
        raise Exception("File Error") from e
    except json.decoder.JSONDecodeError as e:
        raise Exception("JSON Error") from e

    try:
        type_adapter = TypeAdapter(list[dict[str, Any]])
        type_adapter.validate_python(data)

        for fn in data:
            functions.append(FunctionModel(**fn))

    except (AssertionError, KeyError) as e:
        raise Exception("Key Error") from e
    except ValidationError as e:
        raise Exception("Invalid Data") from e

    return functions


if __name__ == "__main__":
    print(parse_functions("./data/input/functions_definition.json"))
