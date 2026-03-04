"""
Agent loop: send user message -> LLM -> if tool_calls then execute tools and append results -> repeat.
"""

import asyncio
import json
from typing import Any, Callable

from openai import OpenAI

from .llm import chat_completion, tools_to_openai


def _message_to_openai(role: str, content: str | list[dict]) -> dict:
    return {"role": role, "content": content}


def _tool_result_message(tool_call_id: str, content: str) -> dict:
    return {"role": "toolResult", "tool_call_id": tool_call_id, "content": content}


async def run_agent_loop(
    client: OpenAI,
    model: str,
    system_prompt: str,
    tools: list[dict],
    user_message: str,
    *,
    history_messages: list[dict] | None = None,
    on_tool_call: Callable[[str, str, dict], Any] | None = None,
    on_event: Callable[[dict[str, Any]], Any] | None = None,
) -> str:
    """
    Run the agent loop until the model returns a final text response (no more tool calls).
    Returns the concatenated text content of the final assistant message.
    
    Args:
        history_messages: Previous conversation messages to include in context.
    """
    name_to_tool = {t["name"]: t for t in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    
    # Add history messages (excluding system messages)
    # history_messages should already include the current user_message if it was saved before calling this function
    if history_messages:
        for msg in history_messages:
            if msg.get("role") != "system":
                messages.append(msg)
    
    # Add current user message only if history is empty or last message is not this user message
    # This handles the case where history_messages is None or doesn't include the current message yet
    if not history_messages or (
        history_messages[-1].get("role") != "user"
        or history_messages[-1].get("content") != user_message
    ):
        messages.append({"role": "user", "content": user_message})

    while True:
        response = chat_completion(client, model, messages, tools, stream=False)
        choice = response.choices[0]
        msg = choice.message
        if not msg.content and not getattr(msg, "tool_calls", None):
            return ""

        if msg.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            messages.append(assistant_msg)
            
            # Emit turn_start event
            if on_event:
                result = on_event({"type": "turn_start"})
                if asyncio.iscoroutine(result):
                    await result
            
            for tc in msg.tool_calls:
                tid = tc.id
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                
                # Emit tool_execution_start
                if on_event:
                    result = on_event({"type": "tool_execution_start", "tool_call_id": tid, "tool_name": name})
                    if asyncio.iscoroutine(result):
                        await result
                
                if on_tool_call:
                    on_tool_call(tid, name, args)
                
                tool = name_to_tool.get(name)
                success = True
                if not tool:
                    content = f"Unknown tool: {name}"
                    success = False
                else:
                    try:
                        result = await tool["execute"](tid, args)
                        parts = [p for p in result if p.get("type") == "text"]
                        content = "\n".join(p.get("text", "") for p in parts)
                    except Exception as e:
                        content = str(e)
                        success = False

                tool_result_message = _tool_result_message(tid, content)
                messages.append(tool_result_message)

                
                # Emit tool_execution_end
                if on_event:
                    result = on_event(({"type": "message_end", "message": tool_result_message}))
                    if asyncio.iscoroutine(result):
                        await result

                    result = on_event({"type": "tool_execution_end", "tool_call_id": tid, "tool_name": name, "success": success})
                    if asyncio.iscoroutine(result):
                        await result
            
            # Emit turn_end event
            if on_event:
                result = on_event({"type": "turn_end", "message": assistant_msg})
                if asyncio.iscoroutine(result):
                    await result
            
            continue

        return (msg.content or "").strip()


def run_agent_loop_sync(
    client: OpenAI,
    model: str,
    system_prompt: str,
    tools: list[dict],
    user_message: str,
    *,
    on_tool_call: Callable[[str, str, dict], Any] | None = None,
) -> str:
    return asyncio.run(
        run_agent_loop(client, model, system_prompt, tools, user_message, on_tool_call=on_tool_call)
    )
