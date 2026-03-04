"""Bash tool - execute shell commands in cwd."""

import os
import subprocess

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail

BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
    },
    "required": ["command"],
}


def create_bash_tool(cwd: str):
    async def execute(tool_call_id: str, arguments: dict) -> list[dict]:
        command = arguments["command"]
        timeout_sec = arguments.get("timeout")
        abs_cwd = os.path.abspath(cwd)
        if not os.path.isdir(abs_cwd):
            raise RuntimeError(f"Working directory does not exist: {abs_cwd}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=abs_cwd,
                capture_output=True,
                text=True,
                timeout=float(timeout_sec) if timeout_sec else None,
            )
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") + (e.stderr or "")
            result = truncate_tail(out, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES)
            msg = result["content"] or "(no output)"
            if result["truncated"]:
                msg += f"\n\n[Showing last {result['output_lines']} lines. Full output truncated.]"
            msg += f"\n\nCommand timed out after {timeout_sec} seconds"
            raise RuntimeError(msg)
        except Exception as e:
            raise RuntimeError(str(e))

        full_output = (result.stdout or "") + (result.stderr or "")
        if not full_output:
            full_output = "(no output)"

        result_tr = truncate_tail(full_output, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES)
        out = result_tr["content"]

        if result_tr["truncated"]:
            start_line = result_tr["total_lines"] - result_tr["output_lines"] + 1
            end_line = result_tr["total_lines"]
            out += f"\n\n[Showing lines {start_line}-{end_line} of {result_tr['total_lines']}. Full output truncated.]"

        if result.returncode != 0:
            out += f"\n\nCommand exited with code {result.returncode}"
            raise RuntimeError(out)

        return [{"type": "text", "text": out}]

    return {
        "name": "bash",
        "description": (
            f"Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. "
            "Optionally provide a timeout in seconds."
        ),
        "parameters": BASH_SCHEMA,
        "execute": execute,
    }
