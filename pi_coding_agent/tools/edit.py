"""Edit file tool - exact text find and replace."""

import os
import re

from .path_utils import resolve_to_cwd

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "oldText": {"type": "string", "description": "Exact text to find and replace (must match exactly)"},
        "newText": {"type": "string", "description": "New text to replace the old text with"},
    },
    "required": ["path", "oldText", "newText"],
}


def _normalize_lf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).replace("\n", "\n")


def _fuzzy_find(content: str, old_text: str) -> tuple[int, int, str] | None:
    """Try exact match first, then normalized. Returns (index, match_length, content_used) or None."""
    if old_text in content:
        idx = content.index(old_text)
        return (idx, len(old_text), content)
    norm_content = _normalize_lf(content)
    norm_old = _normalize_lf(old_text)
    if norm_old in norm_content:
        idx = norm_content.index(norm_old)
        return (idx, len(norm_old), norm_content)
    fuzzy_c = _normalize_whitespace(norm_content)
    fuzzy_o = _normalize_whitespace(norm_old)
    if fuzzy_o not in fuzzy_c:
        return None
    idx = fuzzy_c.index(fuzzy_o)
    return (idx, len(fuzzy_o), fuzzy_c)


def create_edit_tool(cwd: str):
    async def execute(tool_call_id: str, arguments: dict) -> list[dict]:
        path = arguments["path"]
        old_text = arguments["oldText"]
        new_text = arguments["newText"]
        absolute_path = resolve_to_cwd(path, cwd)

        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(absolute_path, "r", encoding="utf-8-sig") as f:
            raw_content = f.read()

        match_result = _fuzzy_find(raw_content, old_text)
        if not match_result:
            raise ValueError(
                f"Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
            )
        idx, match_len, content_for_replacement = match_result
        norm_new = _normalize_lf(new_text)

        needle = content_for_replacement[idx : idx + match_len]
        occurrences = content_for_replacement.count(needle)
        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of the text in {path}. The text must be unique. Provide more context."
            )

        new_content = (
            content_for_replacement[:idx] + norm_new + content_for_replacement[idx + match_len :]
        )
        if content_for_replacement == new_content:
            raise ValueError(
                f"No changes made to {path}. The replacement produced identical content."
            )

        with open(absolute_path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)

        return [{"type": "text", "text": f"Successfully replaced text in {path}."}]

    return {
        "name": "edit",
        "description": "Edit a file by replacing exact text. oldText must match exactly (including whitespace). Use for precise, surgical edits.",
        "parameters": EDIT_SCHEMA,
        "execute": execute,
    }
