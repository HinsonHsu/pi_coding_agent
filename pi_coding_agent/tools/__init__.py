from .bash import create_bash_tool
from .edit import create_edit_tool
from .read import create_read_tool
from .write import create_write_tool


def create_coding_tools(cwd: str) -> list[dict]:
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


__all__ = [
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_read_tool",
    "create_write_tool",
]
