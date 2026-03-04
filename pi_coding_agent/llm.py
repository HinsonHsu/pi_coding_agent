"""OpenAI-compatible LLM client for the agent."""

import json
import os
from typing import Any

from openai import OpenAI


def get_api_key(provider: str | None = None) -> str | None:
    """Resolve API key from env. Prefer OPENAI_API_KEY for default."""
    if provider and provider != "openai":
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
        }
        key = os.environ.get(env_map.get(provider, f"{provider.upper()}_API_KEY"))
        if key:
            return key
    return os.environ.get("OPENAI_API_KEY")


def create_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    key = api_key or get_api_key()
    if not key:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or pass api_key= to create_agent_session."
        )
    return OpenAI(api_key=key, base_url=base_url)


def tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert our tool definitions to OpenAI API format."""
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        })
    return out


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert internal message format to OpenAI API format.
    Specifically converts role="toolResult" to role="tool" with proper structure.
    
    OpenAI API expects:
    - role: "system", "user", "assistant", or "tool"
    - For tool messages: {"role": "tool", "content": str, "tool_call_id": str}
    """
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        
        if role == "toolResult":
            # Convert toolResult to OpenAI's tool format
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                # Fallback: try alternative field names
                tool_call_id = msg.get("toolCallId") or msg.get("tool_call_id")
            
            content = msg.get("content", "")
            # Handle content as string or list of content blocks
            if isinstance(content, list):
                # Extract text content from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block.get("text"), str):
                            text_parts.append(block["text"])
                content = "\n".join(text_parts) if text_parts else ""
            elif not isinstance(content, str):
                content = str(content)
            
            converted.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id,
            })
        elif role in ("system", "user", "assistant"):
            # Pass through standard roles as-is
            converted.append(msg)
        else:
            # Unknown role, skip or convert to user message
            # This handles custom message types that shouldn't be sent to OpenAI
            continue
    
    return converted


def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict],
    *,
    stream: bool = False,
) -> Any:
    """One shot: send messages and optional tool_calls, return completion (or iterator if stream)."""
    # Convert messages to OpenAI API format (toolResult -> tool)
    converted_messages = convert_messages(messages)
    
    openai_tools = tools_to_openai(tools)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": converted_messages,
        "tools": openai_tools if openai_tools else None,
        "stream": stream,
    }
    if not openai_tools:
        kwargs.pop("tools")
    return client.chat.completions.create(**kwargs)


def completion_simple(
    client: OpenAI,
    model: str,
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
) -> str:
    """
    Single completion with no tools. Used for summarization.
    Returns the concatenated text content of the assistant response.
    """
    # Convert messages to OpenAI API format (toolResult -> tool)
    converted_messages = convert_messages(messages)
    
    api_messages: list[dict[str, Any]] = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    api_messages.extend(converted_messages)
    
    response = client.chat.completions.create(
        model=model,
        messages=api_messages,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    msg = choice.message
    if not msg.content:
        return ""
    return (msg.content or "").strip()
