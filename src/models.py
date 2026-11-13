from typing import Any, Self
from enum import StrEnum, auto
from pydantic import BaseModel, Field


class AppError(Exception):
    """Base exception for application-specific errors."""


class JSONType(StrEnum):
    STRING = auto()
    BOOLEAN = auto()
    INTEGER = auto()
    NUMBER = auto()


class TypeModel(BaseModel):
    type: JSONType = Field(alias="type")

    def __repr__(self) -> str:
        return f"{{type: '{self.type}'}}"


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


class EnhancedPrompt(str):
    def __new__(cls, system: str, user: str, assistant: str = "") -> Self:
        value = "\n".join(
            [
                "<|im_start|>system",
                f"{system}<|im_end|>",
                "<|im_start|>user",
                f"{user}<|im_end|>",
                "<|im_start|>assistant",
                assistant,
            ]
        )

        return super().__new__(cls, value)
