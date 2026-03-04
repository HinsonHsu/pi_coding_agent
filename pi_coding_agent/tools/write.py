"""Write file tool."""

import os

from .path_utils import resolve_to_cwd

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


def create_write_tool(cwd: str):
    async def execute(tool_call_id: str, arguments: dict) -> list[dict]:
        path = arguments["path"]
        content = arguments["content"]
        absolute_path = resolve_to_cwd(path, cwd)
        os.makedirs(os.path.dirname(absolute_path) or ".", exist_ok=True)
        with open(absolute_path, "w", encoding="utf-8") as f:
            f.write(content)
        return [{"type": "text", "text": f"Successfully wrote {len(content)} bytes to {path}"}]

    return {
        "name": "write",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories.",
        "parameters": WRITE_SCHEMA,
        "execute": execute,
    }
