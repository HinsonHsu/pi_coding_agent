"""Path resolution relative to cwd; ~ expansion."""

import os
from pathlib import Path


def expand_path(file_path: str) -> str:
    s = file_path.lstrip("@")
    if s == "~":
        return os.path.expanduser("~")
    if s.startswith("~/"):
        return os.path.join(os.path.expanduser("~"), s[2:])
    return s


def resolve_to_cwd(file_path: str, cwd: str) -> str:
    expanded = expand_path(file_path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.normpath(os.path.join(cwd, expanded))


def resolve_read_path(file_path: str, cwd: str) -> str:
    return resolve_to_cwd(file_path, cwd)
