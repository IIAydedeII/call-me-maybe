import json
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

Type = Literal["string", "number"]


class Parameter(BaseModel):
    key: str = Field(min_length=1)
    type_: Type


class Function(BaseModel):
    name: str
    description: str
    parameters: list[Parameter] = Field(min_length=1)
    returns: Type


def parse_functions(file_path: str) -> list[Function]:
    functions: list[Function] = []

    try:
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)

            assert isinstance(data, list)

            for fn in data:
                assert isinstance(fn, dict)
                assert isinstance(fn["parameters"], dict)

                functions.append(
                    Function(
                        name=fn["name"],
                        description=fn["description"],
                        parameters=[
                            Parameter(key=key, type_=value["type"])
                            for key, value in fn["parameters"].items()
                        ],
                        returns=fn["returns"]["type"],
                    )
                )

    except OSError as e:
        raise Exception("File Error") from e
    except json.decoder.JSONDecodeError as e:
        raise Exception("JSON Error") from e
    except (AssertionError, KeyError) as e:
        raise Exception("Key Error") from e
    except ValidationError as e:
        raise Exception("Invalid Data") from e

    return functions


if __name__ == "__main__":
    print(parse_functions("./data/input/functions_definition.json"))
