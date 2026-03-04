# pi-coding-agent (Python)

Python port of the [pi coding-agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent) (TypeScript). Provides an agent that uses **read**, **bash**, **edit**, and **write** tools to fulfill user requests.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt
# Or install as package
pip install -e .

export OPENAI_API_KEY=sk-...

# Interactive mode (default)
pi-py

# Print mode (single-shot)
pi-py -p "What files are in the current directory?"
```

## Features

- **Tools**: read, write, edit, bash (same semantics as the TS version: paths relative to cwd, truncation for read/bash).
- **Interactive mode**: `pi-py` starts a REPL with command history, tool call display, and slash commands.
- **Print mode**: `pi-py -p "prompt"` sends the prompt, runs the agent loop, and prints the final response.
- **SDK**: `create_agent_session()` and `session.prompt(text)` for programmatic use.
- **Session persistence**: `SessionManager` saves conversations to JSONL files in `~/.pi-py/sessions/`.

No TUI, session persistence, extensions, or skills in this version. For the full interactive experience use the TypeScript package `@mariozechner/pi-coding-agent`.

## SDK example

```python
from pi_coding_agent import create_agent_session, SessionManager

# With session persistence (saves to ~/.pi-py/sessions/)
session = create_agent_session(cwd="/path/to/project", model="gpt-4o")

# Subscribe to events
def on_event(event):
    if event["type"] == "tool_execution_start":
        print(f"🔧 Executing: {event['tool_name']}")
    elif event["type"] == "message_end" and event["message"]["role"] == "assistant":
        print(f"✅ Response received")

unsubscribe = session.subscribe(on_event)

# Subscribe to tool calls specifically
session.subscribe_tool_call(lambda tid, name, args: print(f"Tool: {name}", args))

response = session.prompt("List all .py files in src/")
print(response)
print(f"Session file: {session.session_manager.get_session_file()}")

# Unsubscribe when done
unsubscribe()

# In-memory session (no persistence)
session_manager = SessionManager.in_memory()
session = create_agent_session(session_manager=session_manager)

# Cleanup
session.dispose()
```

## CLI

### Interactive Mode (Default)

```bash
pi-py                              # Start interactive REPL
pi-py --model gpt-4o-mini          # Use different model

# Commands in interactive mode:
# /quit, /exit, /q    - Exit
# /clear, /c          - Clear screen
# /help, /h           - Show help
# /model [name]       - Show or set model
# /session            - Show session info
# /cwd                - Show working directory
```

### Print Mode

```bash
pi-py -p "your prompt"             # Single-shot mode
pi-py -p --model gpt-4o-mini "Summarize this repo"
PI_MODEL=gpt-4o-mini pi-py -p "Explain main.py"
```

## Base URL

Use a custom API endpoint (proxy, local model, or other OpenAI-compatible API):

```bash
pi-py -p --base-url https://your-api.com/v1 "Hello"
export OPENAI_BASE_URL=https://your-api.com/v1
pi-py -p "Hello"
```

SDK: `create_agent_session(base_url="https://...")`.

## Requirements

- Python 3.10+
- OpenAI API key (or compatible API via `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`)

## Development

```bash
cd packages/coding-agent-py
pip install -e ".[dev]"
ruff check pi_coding_agent
```
