from __future__ import annotations

"""
Skill loading and formatting (Python port of the TypeScript skills implementation).

This is a minimal implementation of the Agent Skills spec:
- Skills are defined in Markdown files with YAML-like frontmatter.
- Frontmatter keys used here:
  - name: string (optional, falls back to parent directory name)
  - description: string (required, max 1024 chars)
  - disable-model-invocation: boolean (optional)
- Discovery rules:
  - Direct .md children in the skills root directory
  - Recursive SKILL.md files under subdirectories
- Default locations:
  - User skills:   ~/.pi/agent/skills
  - Project skills: <cwd>/.pi/skills
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


@dataclass
class Skill:
    """Loaded skill definition."""

    name: str
    description: str
    file_path: str
    base_dir: str
    source: str  # "user" | "project" | "path"
    disable_model_invocation: bool = False


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse a very small subset of YAML frontmatter.

    Format:
        ---
        key: value
        other: value
        disable-model-invocation: true
        ---
        markdown body...

    This intentionally supports only flat key/value pairs with primitive values.
    If there is no frontmatter block, returns ({}, content).
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    frontmatter_lines: List[str] = []
    end_index = 1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_index = i + 1
            break
        frontmatter_lines.append(lines[i])
    else:
        # No closing '---' found; treat as no frontmatter
        return {}, content

    data: Dict[str, Any] = {}
    for raw in frontmatter_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        lower = value.lower()
        if lower in {"true", "false"}:
            data[key] = lower == "true"
        else:
            data[key] = value

    body = "\n".join(lines[end_index:])
    return data, body


def _validate_name(name: str, parent_dir_name: str) -> List[str]:
    errors: List[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not name or not all(c.islower() or c.isdigit() or c == "-" for c in name):
        errors.append(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
        )
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> List[str]:
    errors: List[str] = []
    if description is None or not description.strip():
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})"
        )
    return errors


def _load_skill_from_file(path: Path, source: str) -> Tuple[Skill | None, List[str]]:
    warnings: List[str] = []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"failed to read skill file {path}: {exc}")
        return None, warnings

    frontmatter, _body = _parse_frontmatter(raw)
    description = str(frontmatter.get("description") or "").strip() or None

    # Validate description first – missing description means we skip the skill entirely.
    warnings.extend(_validate_description(description))
    if description is None:
        return None, warnings

    skill_dir = path.parent
    parent_dir_name = skill_dir.name

    name = str(frontmatter.get("name") or parent_dir_name)
    warnings.extend(_validate_name(name, parent_dir_name))

    disable_model_invocation = bool(
        frontmatter.get("disable-model-invocation") is True
    )

    skill = Skill(
        name=name,
        description=description,
        file_path=str(path.resolve()),
        base_dir=str(skill_dir.resolve()),
        source=source,
        disable_model_invocation=disable_model_invocation,
    )
    return skill, warnings


def _iter_skill_files(root: Path) -> Iterable[Path]:
    """
    Discover candidate skill files under a root directory.

    Rules:
    - Direct .md children in the root directory
    - Recursive SKILL.md under subdirectories
    """
    if not root.is_dir():
        return []

    files: List[Path] = []

    # Root-level .md files
    for child in root.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_file() and child.suffix.lower() == ".md":
            files.append(child)

    # Recursive SKILL.md files
    for skill_md in root.rglob("SKILL.md"):
        if any(part.startswith(".") for part in skill_md.parts):
            # Skip hidden directories like .git
            continue
        files.append(skill_md)

    return files


def load_skills(
    cwd: str,
    *,
    skill_paths: Iterable[str] | None = None,
    include_defaults: bool = True,
) -> Tuple[List[Skill], List[str]]:
    """
    Load skills from all configured locations.

    - User skills:   ~/.pi-py/agent/skills
    - Project skills: <cwd>/.pi/skills
    - Explicit paths: files or directories given in skill_paths

    Returns (skills, warnings).
    """
    warnings: List[str] = []
    skills_by_name: Dict[str, Skill] = {}

    def add_skill(candidate: Skill, source_path: Path) -> None:
        existing = skills_by_name.get(candidate.name)
        if existing:
            warnings.append(
                f'skill name "{candidate.name}" collision between '
                f"{existing.file_path} and {candidate.file_path}; keeping first"
            )
            return
        skills_by_name[candidate.name] = candidate

    cwd_path = Path(cwd).resolve()

    user_skills_root = Path.home() / ".pi-py" / "agent" / "skills"
    project_skills_root = cwd_path / ".pi" / "skills"

    if include_defaults:
        for root, source in (
            (user_skills_root, "user"),
            (project_skills_root, "project"),
        ):
            if not root.exists():
                continue
            for skill_file in _iter_skill_files(root):
                skill, w = _load_skill_from_file(skill_file, source)
                warnings.extend(w)
                if skill is not None:
                    add_skill(skill, skill_file)

    for raw_path in skill_paths or ():
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = cwd_path / path
        try:
            path = path.resolve()
        except OSError:
            # Best-effort resolution
            path = path

        if not path.exists():
            warnings.append(f"skill path does not exist: {path}")
            continue

        if path.is_dir():
            source = "path"
            for skill_file in _iter_skill_files(path):
                skill, w = _load_skill_from_file(skill_file, source)
                warnings.extend(w)
                if skill is not None:
                    add_skill(skill, skill_file)
        elif path.is_file() and path.suffix.lower() == ".md":
            source = "path"
            skill, w = _load_skill_from_file(path, source)
            warnings.extend(w)
            if skill is not None:
                add_skill(skill, path)
        else:
            warnings.append(f"skill path is not a markdown file: {path}")

    return list(skills_by_name.values()), warnings


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(skills: Iterable[Skill]) -> str:
    """
    Format skills for inclusion in a system prompt.

    Uses the same XML shape as the TypeScript implementation so that
    skill files are shared across implementations.

    Skills with disable_model_invocation=True are excluded from the prompt
    (they can only be invoked explicitly via /skill:name commands).
    """
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""

    lines: List[str] = [
        "",
        "",
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill "
        "directory (parent of SKILL.md / dirname of the path) and use that absolute "
        "path in tool commands.",
        "",
        "<available_skills>",
    ]

    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)


__all__ = [
    "Skill",
    "load_skills",
    "format_skills_for_prompt",
]

