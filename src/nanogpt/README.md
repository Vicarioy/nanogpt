# `nanogpt` Python Package

This package contains the installable command-line chatbot code.

- `chatbot.py` classifies simple messages as chat or integer-addition tool calls, validates each intent with JSON Schema, and dispatches only to the allowlisted `add_numbers` tool.
- `__init__.py` provides the package-level `nanogpt` command.

To exercise the tool call from the repository root:

```powershell
.\.venv\Scripts\python.exe .\src\nanogpt\chatbot.py --message "add 12 and 30"
```

Expected response: `12 + 30 = 42 [tool: add_numbers]`.