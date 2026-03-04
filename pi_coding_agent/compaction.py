"""
Context compaction for long sessions - Python version.
References: packages/coding-agent/src/core/compaction/compaction.ts and utils.ts.
"""

from typing import Any

# Default compaction settings
DEFAULT_COMPACTION_SETTINGS = {
    "enabled": True,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000,
}


def estimate_tokens(message: dict[str, Any]) -> int:
    """
    Estimate token count for a message using chars/4 heuristic.
    This is conservative (overestimates tokens).
    """
    chars = 0
    role = message.get("role")

    if role == "user":
        content = message.get("content", "")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chars += len(block.get("text", ""))
    elif role == "assistant":
        content = message.get("content", "")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        chars += len(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        chars += len(block.get("thinking", ""))
                    elif block.get("type") == "toolCall":
                        # Estimate tool call tokens
                        chars += len(str(block))
    elif role == "toolResult":
        content = message.get("content", "")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chars += len(block.get("text", ""))

    return (chars + 3) // 4  # chars/4, rounded up


def estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg)
    # Add overhead for message structure (role, etc.)
    total += len(messages) * 4
    return total


def should_compact(
    context_tokens: int,
    context_window: int,
    settings: dict[str, Any] | None = None,
) -> bool:
    """Check if compaction should trigger based on context usage."""
    if settings is None:
        settings = DEFAULT_COMPACTION_SETTINGS
    if not settings.get("enabled", True):
        return False
    reserve_tokens = settings.get("reserveTokens", 16384)
    return context_tokens > context_window - reserve_tokens


def is_context_overflow(message: dict[str, Any], context_window: int) -> bool:
    """
    Check if message indicates a context overflow error.
    Simplified: checks for common overflow error patterns.
    """
    if message.get("role") != "assistant":
        return False
    error_msg = message.get("errorMessage", "")
    if not error_msg:
        return False
    error_lower = error_msg.lower()
    overflow_keywords = [
        "context length",
        "context window",
        "maximum context length",
        "token limit",
        "too many tokens",
        "context overflow",
    ]
    return any(keyword in error_lower for keyword in overflow_keywords)


# =============================================================================
# Message serialization (for LLM summarization)
# =============================================================================

def serialize_conversation(messages: list[dict[str, Any]]) -> str:
    """
    Serialize messages to text for summarization.
    Prevents the model from treating it as a conversation to continue.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = ""
            if text:
                parts.append(f"[User]: {text}")
        elif role == "assistant":
            content = msg.get("content", "")
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        thinking_parts.append(block.get("thinking", ""))
                    elif block.get("type") == "toolCall":
                        args = block.get("arguments") or {}
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                        tool_calls.append(f"{block.get('name', '')}({args_str})")
            if thinking_parts:
                parts.append(f"[Assistant thinking]: {' '.join(thinking_parts)}")
            if text_parts:
                parts.append(f"[Assistant]: {' '.join(text_parts)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")
        elif role == "toolResult":
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = ""
            if text:
                parts.append(f"[Tool result]: {text}")
    return "\n\n".join(parts)


# =============================================================================
# File operation tracking
# =============================================================================

def extract_file_ops_from_message(message: dict[str, Any], read_files: set[str], modified_files: set[str]) -> None:
    """Extract file paths from tool calls in an assistant message."""
    if message.get("role") != "assistant":
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "toolCall":
            continue
        name = block.get("name")
        args = block.get("arguments")
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not isinstance(path, str):
            continue
        if name == "read":
            read_files.add(path)
        elif name == "write":
            modified_files.add(path)
        elif name == "edit":
            modified_files.add(path)


def compute_file_lists(
    read_files: set[str], modified_files: set[str]
) -> tuple[list[str], list[str]]:
    """Return (read_only_files, modified_files). Read-only = read but not modified."""
    modified = set(modified_files)
    read_only = sorted(read_files - modified)
    modified_list = sorted(modified)
    return (read_only, modified_list)


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Format file operations as XML tags for summary."""
    sections: list[str] = []
    nl = "\n"
    if read_files:
        sections.append(f"<read-files>\n{nl.join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n{nl.join(modified_files)}\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# =============================================================================
# Summarization prompts (from TS compaction.ts / utils.ts)
# =============================================================================

SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


# =============================================================================
# LLM summarization
# =============================================================================

def generate_summary(
    messages_to_summarize: list[dict[str, Any]],
    client: Any,
    model: str,
    reserve_tokens: int,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
) -> str:
    """
    Generate a summary of the conversation using the LLM.
    If previous_summary is provided, uses the update prompt to merge.
    """
    from .llm import completion_simple

    max_tokens = int(0.8 * reserve_tokens)
    max_tokens = max(512, min(max_tokens, 8192))

    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    conversation_text = serialize_conversation(messages_to_summarize)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    summarization_messages = [
        {"role": "user", "content": prompt_text},
    ]

    response_text = completion_simple(
        client,
        model,
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=summarization_messages,
        max_tokens=max_tokens,
    )
    return response_text
