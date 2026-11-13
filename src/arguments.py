"""Configure and build the command-line argument parser."""

from argparse import ArgumentParser
from pathlib import Path


def build_parser() -> ArgumentParser:
    """Build and configure the command-line argument parser.

    Each argument has a default path that is used when no custom path
    is provided by the user.

    Returns:
        ArgumentParser: The configured argument parser.
    """
    parser = ArgumentParser(
        description="Pick a function with the help of a language model "
        "and output the predicted usage.",
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="File containing function definitions",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="File containing the prompts to process.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="File to output the predicted usages to.",
    )

    return parser
