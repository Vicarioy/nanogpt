import unittest

from jsonschema import ValidationError

from src.nanogpt.chatbot import (
    INTENT_VALIDATOR,
    classify_intent,
    dispatch_intent,
    respond,
)


class ChatbotTests(unittest.TestCase):
    def test_add_tool_call_runs_end_to_end(self):
        intent = classify_intent("add 12 and 30")
        INTENT_VALIDATOR.validate(intent)
        self.assertEqual(dispatch_intent(intent), "12 + 30 = 42 [tool: add_numbers]")

    def test_invalid_tool_intent_is_rejected_by_schema(self):
        invalid_intent = {
            "intent": "tool_call",
            "tool": "run_arbitrary_code",
            "arguments": {"left": 1, "right": 2},
        }
        with self.assertRaises(ValidationError):
            dispatch_intent(invalid_intent)

    def test_regular_message_gets_chat_response(self):
        self.assertIn("add two whole numbers", respond("hello"))


if __name__ == "__main__":
    unittest.main()
