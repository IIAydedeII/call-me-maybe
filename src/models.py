from typing import Any
from enum import StrEnum, auto
from pydantic import BaseModel, Field


class JSONType(StrEnum):
    STRING = auto()
    BOOLEAN = auto()
    INTEGER = auto()
    NUMBER = auto()


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


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
