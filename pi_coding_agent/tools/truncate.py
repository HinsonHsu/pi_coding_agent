"""
Truncation utilities for tool outputs.
Limits: lines (default 2000) and bytes (default 50KB).
"""

from typing import Literal

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB


def format_size(bytes_val: int) -> str:
    if bytes_val < 1024:
        return f"{bytes_val}B"
    if bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f}KB"
    return f"{bytes_val / (1024 * 1024):.1f}MB"


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    """Keep first N lines/bytes. Never returns partial lines."""
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return {
            "content": content,
            "truncated": False,
            "truncated_by": None,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "output_lines": total_lines,
            "output_bytes": total_bytes,
            "first_line_exceeds_limit": False,
        }

    first_line_bytes = len(lines[0].encode("utf-8")) if lines else 0
    if first_line_bytes > max_bytes:
        return {
            "content": "",
            "truncated": True,
            "truncated_by": "bytes",
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "output_lines": 0,
            "output_bytes": 0,
            "first_line_exceeds_limit": True,
        }

    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for i, line in enumerate(lines):
        if i >= max_lines:
            truncated_by = "lines"
            break
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines_arr.append(line)
        output_bytes_count += line_bytes

    output_content = "\n".join(output_lines_arr)
    return {
        "content": output_content,
        "truncated": True,
        "truncated_by": truncated_by,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "output_lines": len(output_lines_arr),
        "output_bytes": len(output_content.encode("utf-8")),
        "first_line_exceeds_limit": False,
    }


def _truncate_string_to_bytes_from_end(s: str, max_bytes: int) -> str:
    buf = s.encode("utf-8")
    if len(buf) <= max_bytes:
        return s
    start = len(buf) - max_bytes
    while start < len(buf) and (buf[start] & 0xC0) == 0x80:
        start += 1
    return buf[start:].decode("utf-8")


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    """Keep last N lines/bytes. May return partial first line for bash tail."""
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return {
            "content": content,
            "truncated": False,
            "truncated_by": None,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "output_lines": total_lines,
            "output_bytes": total_bytes,
            "last_line_partial": False,
        }

    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(output_lines_arr) >= max_lines:
            truncated_by = "lines"
            break
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if output_lines_arr else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines_arr:
                output_lines_arr.insert(0, _truncate_string_to_bytes_from_end(line, max_bytes))
                output_bytes_count = len(output_lines_arr[0].encode("utf-8"))
                last_line_partial = True
            break
        output_lines_arr.insert(0, line)
        output_bytes_count += line_bytes

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    return {
        "content": output_content,
        "truncated": True,
        "truncated_by": truncated_by,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "output_lines": len(output_lines_arr),
        "output_bytes": len(output_content.encode("utf-8")),
        "last_line_partial": last_line_partial,
    }
