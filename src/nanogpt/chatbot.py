"""A small schema-validated intent router with a safe arithmetic tool."""

import argparse
import re

from jsonschema import Draft202012Validator


INTENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "intent": {"const": "chat"},
                "message": {"type": "string", "minLength": 1},
            },
            "required": ["intent", "message"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "intent": {"const": "tool_call"},
                "tool": {"const": "add_numbers"},
                "arguments": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "integer"},
                        "right": {"type": "integer"},
                    },
                    "required": ["left", "right"],
                    "additionalProperties": False,
                },
            },
            "required": ["intent", "tool", "arguments"],
            "additionalProperties": False,
        },
    ],
}
INTENT_VALIDATOR = Draft202012Validator(INTENT_SCHEMA)
ADD_PATTERN = re.compile(r"(?:add\s+)?(-?\d+)\s*(?:and|\+)\s*(-?\d+)", re.I)


def add_numbers(left: int, right: int) -> int:
    return left + right


TOOLS = {"add_numbers": add_numbers}


def classify_intent(message: str) -> dict:
    """Convert the supported addition syntax or ordinary text into an intent."""
    match = ADD_PATTERN.fullmatch(message.strip())
    if match:
        return {
            "intent": "tool_call",
            "tool": "add_numbers",
            "arguments": {"left": int(match.group(1)), "right": int(match.group(2))},
        }
    return {"intent": "chat", "message": message.strip() or "hello"}


def dispatch_intent(intent: dict) -> str:
    """Validate an intent before replying or dispatching an allowlisted tool."""
    INTENT_VALIDATOR.validate(intent)
    if intent["intent"] == "chat":
        return "I can add two whole numbers. Try: add 12 and 30."

    tool = TOOLS[intent["tool"]]
    arguments = intent["arguments"]
    result = tool(arguments["left"], arguments["right"])
    return f"{arguments['left']} + {arguments['right']} = {result} [tool: {intent['tool']}]"


def respond(message: str) -> str:
    return dispatch_intent(classify_intent(message))


def main() -> None:
    parser = argparse.ArgumentParser(description="Schema-validated intent chatbot")
    parser.add_argument("--message", help="Handle one message, then exit")
    args = parser.parse_args()

    if args.message is not None:
        print(respond(args.message))
        return

    print("Chatbot ready. Enter 'quit' to exit.")
    while True:
        try:
            message = input("you> ").strip()
        except EOFError:
            break
        if message.lower() in {"quit", "exit"}:
            break
        if message:
            print(f"bot> {respond(message)}")


if __name__ == "__main__":
    main()
