# Tests

This folder contains focused regression tests for chatbot intent validation/tool dispatch and training fit diagnostics.

Run the tests from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The chatbot tests confirm the addition tool works end-to-end and invalid tool names are rejected by schema validation. The training tests exercise the overfitting, underfitting, and insufficient-history diagnostic paths.