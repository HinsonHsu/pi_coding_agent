"""Read file tool - text and optional image support."""

from .path_utils import resolve_read_path
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "integer", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}


def create_read_tool(cwd: str):
    max_lines = DEFAULT_MAX_LINES
    max_bytes = DEFAULT_MAX_BYTES
    desc = (
        f"Read the contents of a file. For text files, output is truncated to {max_lines} lines or "
        f"{max_bytes // 1024}KB. Use offset/limit for large files."
    )

    async def execute(tool_call_id: str, arguments: dict) -> list[dict]:
        path = arguments["path"]
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        absolute_path = resolve_read_path(path, cwd)

        try:
            with open(absolute_path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")

        try:
            text_content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return [{"type": "text", "text": f"File is binary or not UTF-8: {path}"}]

        lines = text_content.split("\n")
        total_file_lines = len(lines)
        start_line = (offset - 1) if offset else 0
        start_line = max(0, start_line)

        if start_line >= len(lines):
            raise ValueError(f"Offset {offset} is beyond end of file ({len(lines)} lines total)")

        if limit is not None:
            end_line = min(start_line + limit, len(lines))
            selected = "\n".join(lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(lines[start_line:])
            user_limited_lines = None

        result = truncate_head(selected, max_lines=max_lines, max_bytes=max_bytes)
        out = result["content"]

        if result.get("first_line_exceeds_limit"):
            start_display = start_line + 1
            return [
                {
                    "type": "text",
                    "text": f"[Line {start_display} exceeds {format_size(max_bytes)} limit. Use bash: sed -n '{start_display}p' {path} | head -c {max_bytes}]",
                }
            ]
        if result["truncated"]:
            start_display = start_line + 1
            end_display = start_display + result["output_lines"] - 1
            next_offset = end_display + 1
            by = result["truncated_by"]
            if by == "lines":
                out += f"\n\n[Showing lines {start_display}-{end_display} of {total_file_lines}. Use offset={next_offset} to continue.]"
            else:
                out += f"\n\n[Showing lines {start_display}-{end_display} of {total_file_lines} ({format_size(max_bytes)} limit). Use offset={next_offset} to continue.]"
        elif user_limited_lines is not None and start_line + user_limited_lines < len(lines):
            next_offset = start_line + user_limited_lines + 1
            remaining = len(lines) - (start_line + user_limited_lines)
            out += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"

        return [{"type": "text", "text": out}]

    return {
        "name": "read",
        "description": desc,
        "parameters": READ_SCHEMA,
        "execute": execute,
    }
