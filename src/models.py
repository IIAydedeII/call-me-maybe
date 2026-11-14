"""Define data models and prompt representations for function calling."""

from __future__ import annotations
from typing import Any
from enum import Enum
from pydantic import BaseModel, Field


class JSONType(str, Enum):
    """Represent supported JSON data types."""

    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"


class TypeModel(BaseModel):
    """Represent the expected type of a function parameter or the return value.

    Attributes:
        type (JSONType):
            The expected JSON data type, serialized and validated using the
            field alias ``type``.
    """

    type: JSONType = Field(alias="type")

    def __repr__(self) -> str:
        return f"{{type: '{self.type}'}}"


class FunctionModel(BaseModel):
    """Represent a function definition and its expected interface.

    Attributes:
        name (str): The function name. Must contain at least one character.
        description (str):
            A natural language description of the function's purpose or
            behavior.
        parameters (dict[str, TypeModel]):
            A mapping of parameter names to their expected data types.
        returns (TypeModel): The expected return data type of the function.
    """

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, TypeModel]
    returns: TypeModel


class PromptModel(BaseModel):
    """Represent a user prompt to be processed.

    Attributes:
        prompt (str): The natural language request.
    """

    prompt: str


class FunctionCall(BaseModel):
    """Represent a generated function call.

    Attributes:
        prompt (str):
            The original user request associated with the function call.
        name (str): The name of the selected function.
        parameters (dict[str, Any]):
            A mapping of parameter names to their generated values.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]


class EnhancedPrompt(str):
    """Represent a formatted conversational prompt for a language model.

    Constructs a prompt containing system, user, and assistant messages using
    the model's chat format.
    """

    def __new__(
        cls, system: str, user: str, assistant: str = ""
    ) -> EnhancedPrompt:
        """Construct a formatted conversational prompt.

        Combines the system instructions, user request, and optional
        assistant prefix into a single string using chat message tokens.

        Args:
            system (str):
                The system message containing instructions for the model.
            user (str): The user message containing the request or input data.
            assistant (str):
                An optional assistant message prefix used to guide or continue
                the model's generation.

        Returns:
            EnhancedPrompt: Contains the formatted conversational prompt.
        """
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
