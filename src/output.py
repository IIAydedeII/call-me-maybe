import json
from pathlib import Path
from dataclasses import dataclass, asdict


class OutputError(Exception):
    """Raised when the data cannot be written to the file in JSON format."""


@dataclass
class FunctionCall:
    prompt: str
    name: str
    parameters: dict[str, str]


def output_json(file_path: str, data: list[FunctionCall]) -> None:
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            data=json.dumps([asdict(data) for data in data], indent=4),
            encoding="utf-8",
        )
    except OSError as e:
        raise OutputError("Unable to write to the specified file") from e
    except json.JSONDecodeError as e:
        raise OutputError("Cannot serialize the data as JSON") from e
