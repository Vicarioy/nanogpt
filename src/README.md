# Source Package

This folder uses the `src` layout for installable Python package code. The `nanogpt` package lives in `src/nanogpt/` and provides the project's command-line chatbot entry point.

From the repository root, run the chatbot directly with:

```powershell
.\.venv\Scripts\python.exe .\src\nanogpt\chatbot.py --message "add 12 and 30"
```

The project also registers the `nanogpt-chat` console command when installed in the active environment.