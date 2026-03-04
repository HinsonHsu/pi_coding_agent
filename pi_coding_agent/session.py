"""
Agent session: holds cwd, tools, LLM client, system prompt; exposes prompt().
"""

import asyncio
import os
from typing import Any, Callable

from openai import OpenAI

from .agent_loop import run_agent_loop
from .compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    compute_file_lists,
    estimate_context_tokens,
    estimate_tokens,
    extract_file_ops_from_message,
    format_file_operations,
    generate_summary,
    is_context_overflow,
    should_compact,
)
from .llm import create_client
from .session_manager import SessionManager
from .system_prompt import build_system_prompt
from .tools import create_coding_tools

# Event types for subscription (mirrors TypeScript AgentSessionEvent concept)
AgentSessionEvent = dict[str, Any]
AgentSessionEventListener = Callable[[AgentSessionEvent], None]


def _entries_to_messages(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert branch entries to message dicts (for token estimation and summarization)."""
    out: list[dict[str, Any]] = []
    for e in entries:
        t = e.get("type")
        if t == "message":
            m = e.get("message")
            if m:
                out.append(m)
        elif t == "custom_message":
            out.append({"role": "user", "content": e.get("content", "")})
        elif t == "branch_summary":
            out.append({"role": "user", "content": f"[Branch summary: {e.get('summary', '')}]"})
        elif t == "compaction":
            out.append({"role": "user", "content": f"[Previous conversation summarized: {e.get('summary', '')}]"})
    return out


class AgentSession:
    """
    Session for the coding agent. Use prompt(text) to send a message and get the final response.
    """

    def __init__(
        self,
        *,
        cwd: str | None = None,
        model: str = "gpt-4o",
        client: OpenAI | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        tools: list[dict] | None = None,
        system_prompt: str | None = None,
        session_manager: SessionManager | None = None,
    ):
        self._cwd = os.path.abspath(cwd or os.getcwd())
        self._model = model
        self._client = client or create_client(api_key=api_key, base_url=base_url)
        self._tools = tools if tools is not None else create_coding_tools(self._cwd)
        self._system_prompt = system_prompt or build_system_prompt(
            self._cwd,
            selected_tools=[t["name"] for t in self._tools],
        )
        self._session_manager = session_manager or SessionManager.in_memory(self._cwd)
        self._on_tool_call: Callable[[str, str, dict], None] | None = None

        # Event subscription system
        self._event_listeners: list[AgentSessionEventListener] = []
        self._unsubscribe_agent: Callable[[], None] | None = None
        
        # Compaction state
        self._last_assistant_message: dict[str, Any] | None = None
        self._compaction_settings = DEFAULT_COMPACTION_SETTINGS.copy()

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def model(self) -> str:
        return self._model

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    def set_model(self, model: str) -> None:
        self._model = model
        self._session_manager.append_model_change("openai", model)

    def subscribe_tool_call(self, callback: Callable[[str, str, dict], None]) -> None:
        """Call callback(tool_call_id, tool_name, arguments) for each tool execution."""
        self._on_tool_call = callback

    # -------------------------------------------------------------------------
    # Event subscription (public)
    # -------------------------------------------------------------------------

    def subscribe(self, listener: AgentSessionEventListener) -> Callable[[], None]:
        """
        Subscribe to agent session events.
        Multiple listeners can be added. Returns unsubscribe function for this listener.
        
        Event types:
        - {"type": "message_start", "message": {...}}
        - {"type": "message_end", "message": {...}}
        - {"type": "tool_execution_start", "tool_call_id": "...", "tool_name": "..."}
        - {"type": "tool_execution_end", "tool_call_id": "...", "tool_name": "...", "success": bool}
        - {"type": "agent_start"}
        - {"type": "agent_end", "messages": [...]}
        """
        self._event_listeners.append(listener)
        
        # Return unsubscribe function for this specific listener
        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)
        
        return unsubscribe
    # -------------------------------------------------------------------------
    # Internal event handling (similar to TS _handleAgentEvent)
    # -------------------------------------------------------------------------

    def _emit(self, event: AgentSessionEvent) -> None:
        """Emit an event to all listeners."""
        for listener in list(self._event_listeners):
            try:
                listener(event)
            except Exception:
                # Listener errors must not break the agent
                continue

    async def _handle_agent_event(self, event: AgentSessionEvent) -> None:
        """
        Internal handler for agent events.

        Responsibilities (simplified vs. TypeScript version):
        - Forward events to subscribers.
        - Persist messages to SessionManager on message_end.
        - Track last assistant message for auto-compaction check.
        - Check compaction on agent_end.
        """
        event_type = event.get("type")

        # Session persistence on message_end
        if event_type == "message_end":
            msg = event.get("message")
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant", "toolResult"):
                self._session_manager.append_message(msg)
            
            # Track assistant message for auto-compaction (checked on agent_end)
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                self._last_assistant_message = msg

        # Check auto-compaction after agent completes (mirrors TypeScript _handleAgentEvent)
        if event_type == "agent_end" and self._last_assistant_message:
            msg = self._last_assistant_message
            self._last_assistant_message = None
            
            # Check for retryable errors first (simplified: skip for now)
            # In full version: if self._is_retryable_error(msg):
            #     did_retry = await self._handle_retryable_error(msg)
            #     if did_retry:
            #         return  # Retry was initiated, don't proceed to compaction
            
            await self._check_compaction(msg)

        # Forward to user listeners
        self._emit(event)

    # -------------------------------------------------------------------------
    # Agent wiring / lifecycle
    # -------------------------------------------------------------------------

    def _emit(self, event: AgentSessionEvent) -> None:
        """Backward-compatible alias kept for existing call sites."""
        for listener in list(self._event_listeners):
            try:
                listener(event)
            except Exception:
                continue

    def _disconnect_from_agent(self) -> None:
        """Temporarily disconnect from agent events. User listeners are preserved."""
        if self._unsubscribe_agent:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None

    def _reconnect_to_agent(self) -> None:
        """Reconnect to agent events after _disconnect_from_agent()."""
        if self._unsubscribe_agent:
            return  # Already connected
        # In Python version, we reconnect by re-subscribing in the next prompt call
        # This is simpler than TypeScript version which has a persistent agent subscription
        pass

    def dispose(self) -> None:
        """
        Remove all listeners and disconnect from agent.
        Call this when completely done with the session.
        """
        self._disconnect_from_agent()
        self._event_listeners.clear()

    # -------------------------------------------------------------------------
    # Compaction
    # -------------------------------------------------------------------------

    def set_compaction_enabled(self, enabled: bool) -> None:
        """Toggle auto-compaction setting."""
        self._compaction_settings["enabled"] = enabled

    @property
    def compaction_enabled(self) -> bool:
        """Whether auto-compaction is enabled."""
        return self._compaction_settings.get("enabled", True)

    async def _check_compaction(self, assistant_message: dict[str, Any]) -> None:
        """
        Check if compaction is needed and run it.
        
        Two cases:
        1. Overflow: LLM returned context overflow error, compact and auto-retry
        2. Threshold: Context over threshold, compact (no auto-retry)
        """
        if not self._compaction_settings.get("enabled", True):
            return

        # Skip if message was aborted (user cancelled)
        if assistant_message.get("stopReason") == "aborted":
            return

        # Default context window (128k tokens for most models)
        context_window = 128000

        # Case 1: Overflow - LLM returned context overflow error
        if is_context_overflow(assistant_message, context_window):
            await self._run_auto_compaction("overflow", will_retry=False)  # Simplified: no auto-retry in Python version
            return

        # Case 2: Threshold - turn succeeded but context is getting large
        # Skip if this was an error (non-overflow errors don't have usage data)
        if assistant_message.get("stopReason") == "error":
            return

        # Estimate context tokens from session messages
        context = self._session_manager.build_session_context()
        messages = context.get("messages", [])
        context_tokens = estimate_context_tokens(messages)

        if should_compact(context_tokens, context_window, self._compaction_settings):
            await self._run_auto_compaction("threshold", will_retry=False)

    async def _run_auto_compaction(self, reason: str, will_retry: bool) -> None:
        """
        Internal: Run auto-compaction with events.
        Simplified version - just logs for now since full compaction requires LLM summarization.
        """
        self._emit({"type": "auto_compaction_start", "reason": reason})

        try:
            branch = self._session_manager.get_branch()
            if branch and branch[-1].get("type") == "compaction":
                self._emit({"type": "auto_compaction_end", "result": None, "aborted": False, "willRetry": False})
                return

            # Find last compaction index
            prev_compaction_index = -1
            for i in range(len(branch) - 1, -1, -1):
                if branch[i].get("type") == "compaction":
                    prev_compaction_index = i
                    break
            boundary_start = prev_compaction_index + 1
            boundary_end = len(branch)
            usage_start = prev_compaction_index if prev_compaction_index >= 0 else 0
            usage_entries = branch[usage_start:boundary_end]
            tokens_before = estimate_context_tokens(_entries_to_messages(usage_entries))

            keep_recent_tokens = self._compaction_settings.get("keepRecentTokens", 20000)
            reserve_tokens = self._compaction_settings.get("reserveTokens", 16384)

            # Find cut point: walk backwards, accumulate tokens until >= keep_recent_tokens
            accumulated = 0
            first_kept_entry_index = boundary_start
            for i in range(boundary_end - 1, boundary_start - 1, -1):
                entry = branch[i]
                if entry.get("type") != "message":
                    continue
                msg = entry.get("message")
                if not msg:
                    continue
                accumulated += estimate_tokens(msg)
                if accumulated >= keep_recent_tokens:
                    first_kept_entry_index = i
                    break

            if first_kept_entry_index <= boundary_start:
                self._emit({"type": "auto_compaction_end", "result": None, "aborted": False, "willRetry": False})
                return

            first_kept_entry_id = branch[first_kept_entry_index].get("id")
            if not first_kept_entry_id:
                self._emit({"type": "auto_compaction_end", "result": None, "aborted": False, "willRetry": False})
                return

            messages_to_summarize = _entries_to_messages(branch[boundary_start:first_kept_entry_index])
            if not messages_to_summarize:
                self._emit({"type": "auto_compaction_end", "result": None, "aborted": False, "willRetry": False})
                return

            previous_summary: str | None = None
            if prev_compaction_index >= 0:
                previous_summary = branch[prev_compaction_index].get("summary")

            read_files: set[str] = set()
            modified_files: set[str] = set()
            if prev_compaction_index >= 0:
                details = branch[prev_compaction_index].get("details") or {}
                read_files.update(details.get("readFiles") or [])
                modified_files.update(details.get("modifiedFiles") or [])
            for msg in messages_to_summarize:
                extract_file_ops_from_message(msg, read_files, modified_files)

            summary = generate_summary(
                messages_to_summarize,
                self._client,
                self._model,
                reserve_tokens,
                previous_summary=previous_summary,
            )
            read_only_list, modified_list = compute_file_lists(read_files, modified_files)
            summary += format_file_operations(read_only_list, modified_list)

            self._session_manager.append_compaction(
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
                details={"readFiles": read_only_list, "modifiedFiles": modified_list},
            )

            result = {
                "summary": summary,
                "firstKeptEntryId": first_kept_entry_id,
                "tokensBefore": tokens_before,
            }
            self._emit({"type": "auto_compaction_end", "result": result, "aborted": False, "willRetry": will_retry})

        except Exception as e:
            error_message = str(e) if isinstance(e, Exception) else "compaction failed"
            self._emit({
                "type": "auto_compaction_end",
                "result": None,
                "aborted": False,
                "willRetry": False,
                "errorMessage": (
                    f"Context overflow recovery failed: {error_message}"
                    if reason == "overflow"
                    else f"Auto-compaction failed: {error_message}"
                ),
            })

    async def prompt_async(self, text: str) -> str:
        """Send a user message and return the final assistant text response."""
        # Emit agent_start event
        await self._handle_agent_event({"type": "agent_start"})

        # User message event (persistence happens on message_end)
        user_message = {"role": "user", "content": text}
        await self._handle_agent_event({"type": "message_start", "message": user_message})
        await self._handle_agent_event({"type": "message_end", "message": user_message})

        # Build session context to get history messages (now includes the user message we just added)
        context = self._session_manager.build_session_context()
        history_messages = context.get("messages", [])

        # No long-lived subscription in Python version; keep for API parity
        self._unsubscribe_agent = lambda: None

        # Run agent loop with history; events flow through _handle_agent_event
        response = await run_agent_loop(
            self._client,
            self._model,
            self._system_prompt,
            self._tools,
            text,
            history_messages=history_messages,
            on_tool_call=self._on_tool_call,
            on_event=self._handle_agent_event,
        )

        # Assistant message event (persistence happens on message_end)
        assistant_message = {"role": "assistant", "content": response}
        await self._handle_agent_event({"type": "message_start", "message": assistant_message})
        await self._handle_agent_event({"type": "message_end", "message": assistant_message})

        # Emit agent_end event (this will trigger _check_compaction inside _handle_agent_event)
        await self._handle_agent_event({"type": "agent_end", "messages": [user_message, assistant_message]})

        # Clear unsubscribe after completion
        self._unsubscribe_agent = None

        return response

    def prompt(self, text: str) -> str:
        """Send a user message and return the final assistant text response (sync)."""
        return asyncio.run(self.prompt_async(text))

    async def compact_async(self, custom_instructions: str | None = None) -> dict[str, Any]:
        """
        Manually trigger compaction with optional custom instructions.
        Returns CompactionResult with summary, firstKeptEntryId, tokensBefore.
        """
        branch = self._session_manager.get_branch()
        if not branch:
            raise ValueError("Nothing to compact (no session)")
        
        # Check if already compacted
        if branch[-1].get("type") == "compaction":
            raise ValueError("Already compacted")
        
        # Check if there are enough messages
        message_count = sum(1 for e in branch if e.get("type") == "message")
        if message_count < 2:
            raise ValueError("Nothing to compact (need at least 2 messages)")
        
        # Use the same logic as _run_auto_compaction but with custom instructions
        prev_compaction_index = -1
        for i in range(len(branch) - 1, -1, -1):
            if branch[i].get("type") == "compaction":
                prev_compaction_index = i
                break
        boundary_start = prev_compaction_index + 1
        boundary_end = len(branch)
        usage_start = prev_compaction_index if prev_compaction_index >= 0 else 0
        usage_entries = branch[usage_start:boundary_end]
        tokens_before = estimate_context_tokens(_entries_to_messages(usage_entries))

        keep_recent_tokens = self._compaction_settings.get("keepRecentTokens", 20000)
        reserve_tokens = self._compaction_settings.get("reserveTokens", 16384)

        # Find cut point: walk backwards, accumulate tokens until >= keep_recent_tokens
        accumulated = 0
        first_kept_entry_index = boundary_start
        for i in range(boundary_end - 1, boundary_start - 1, -1):
            entry = branch[i]
            if entry.get("type") != "message":
                continue
            msg = entry.get("message")
            if not msg:
                continue
            accumulated += estimate_tokens(msg)
            if accumulated >= keep_recent_tokens:
                first_kept_entry_index = i
                break

        if first_kept_entry_index <= boundary_start:
            raise ValueError(f"Nothing to compact (session too small),{first_kept_entry_index}<={boundary_start}")

        first_kept_entry_id = branch[first_kept_entry_index].get("id")
        if not first_kept_entry_id:
            raise ValueError("First kept entry has no ID")

        messages_to_summarize = _entries_to_messages(branch[boundary_start:first_kept_entry_index])
        if not messages_to_summarize:
            raise ValueError("Nothing to compact (no messages to summarize)")

        previous_summary: str | None = None
        if prev_compaction_index >= 0:
            previous_summary = branch[prev_compaction_index].get("summary")

        read_files: set[str] = set()
        modified_files: set[str] = set()
        if prev_compaction_index >= 0:
            details = branch[prev_compaction_index].get("details") or {}
            read_files.update(details.get("readFiles") or [])
            modified_files.update(details.get("modifiedFiles") or [])
        for msg in messages_to_summarize:
            extract_file_ops_from_message(msg, read_files, modified_files)

        summary = generate_summary(
            messages_to_summarize,
            self._client,
            self._model,
            reserve_tokens,
            custom_instructions=custom_instructions,
            previous_summary=previous_summary,
        )
        read_only_list, modified_list = compute_file_lists(read_files, modified_files)
        summary += format_file_operations(read_only_list, modified_list)

        self._session_manager.append_compaction(
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details={"readFiles": read_only_list, "modifiedFiles": modified_list},
        )

        return {
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        }

    def compact(self, custom_instructions: str | None = None) -> dict[str, Any]:
        """Manually trigger compaction (sync wrapper)."""
        return asyncio.run(self.compact_async(custom_instructions))

    def switch_session(self, session_path: str) -> None:
        """
        Switch to a different session file (used for resume).
        Reloads messages from the new session.
        """
        # Disconnect from any running agent
        if self._unsubscribe_agent:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None
        
        # Set new session file
        self._session_manager.set_session_file(session_path)
        
        # Reload messages (session manager handles this internally)
        # The next prompt() call will use the new session's messages


def create_agent_session(
    *,
    cwd: str | None = None,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    client: OpenAI | None = None,
    tools: list[dict] | None = None,
    system_prompt: str | None = None,
    session_manager: SessionManager | None = None,
) -> AgentSession:
    """
    Create an agent session with default tools (read, bash, edit, write).
    Set OPENAI_API_KEY or pass api_key= to authenticate.
    Pass base_url= for OpenAI-compatible endpoints (e.g. proxy, local model).
    Pass session_manager= to override session persistence (default: file-based in ~/.pi-py/sessions/).
    """
    resolved_cwd = os.path.abspath(cwd or os.getcwd())
    if session_manager is None:
        session_manager = SessionManager.create(resolved_cwd)
    session = AgentSession(
        cwd=resolved_cwd,
        model=model,
        client=client,
        api_key=api_key,
        base_url=base_url,
        tools=tools,
        system_prompt=system_prompt,
        session_manager=session_manager,
    )
    # Save initial model
    session_manager.append_model_change("openai", model)
    return session
