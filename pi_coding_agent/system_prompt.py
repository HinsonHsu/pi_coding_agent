"""System prompt for the coding agent."""

from datetime import datetime

from .skill import format_skills_for_prompt, load_skills


def build_system_prompt(
    cwd: str,
    *,
    selected_tools: list[str] | None = None,
    include_skills: bool = True,
    skill_paths: list[str] | None = None,
) -> str:
    tools = selected_tools or ["read", "bash", "edit", "write"]
    tool_descriptions = {
        "read": "Read file contents",
        "bash": "Execute bash commands (ls, grep, find, etc.)",
        "edit": "Make surgical edits to files (find exact text and replace)",
        "write": "Create or overwrite files",
    }
    tools_list = "\n".join(f"- {t}: {tool_descriptions.get(t, t)}" for t in tools if t in tool_descriptions)
    if not tools_list:
        tools_list = "(none)"

    guidelines = [
        "Use read to examine files before editing. You must use this tool instead of cat or sed.",
        "Use edit for precise changes (old text must match exactly).",
        "Use write only for new files or complete rewrites.",
        "When summarizing your actions, output plain text directly - do NOT use cat or bash to display what you did.",
        "Be concise in your responses.",
        "Show file paths clearly when working with files.",
    ]
    guidelines_text = "\n".join(f"- {g}" for g in guidelines)

    skills_text = ""
    if include_skills:
        skills, _warnings = load_skills(cwd, skill_paths=skill_paths)
        skills_text = format_skills_for_prompt(skills)

    now = datetime.now().strftime("%A, %B %d, %Y, %I:%M:%S %p %Z")
    base = f"""You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

Guidelines:
{guidelines_text}

Current date and time: {now}
Current working directory: {cwd}"""
    if skills_text:
        return base + skills_text
    return base
