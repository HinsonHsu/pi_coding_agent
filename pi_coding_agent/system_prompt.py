"""System prompt for the coding agent."""

from datetime import datetime


def build_system_prompt(
    cwd: str,
    *,
    selected_tools: list[str] | None = None,
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

    now = datetime.now().strftime("%A, %B %d, %Y, %I:%M:%S %p %Z")
    return f"""You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

Guidelines:
{guidelines_text}

Current date and time: {now}
Current working directory: {cwd}"""
