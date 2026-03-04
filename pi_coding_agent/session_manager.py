"""
Session manager for persisting agent conversations to JSONL files.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT_SESSION_VERSION = 3


def _get_sessions_base_dir(agent_dir: str | None = None) -> str:
    """Get base sessions directory (~/.pi-py/sessions)."""
    if agent_dir is None:
        agent_dir = os.path.expanduser("~/.pi-py")
    return os.path.join(agent_dir, "sessions")


def _extract_text_content(message: dict[str, Any]) -> str:
    """Extract text content from a message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return " ".join(text_parts)
    return ""


def _is_message_with_content(message: dict[str, Any]) -> bool:
    """Check if message has content."""
    content = message.get("content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip()
            for block in content
        )
    return False


def _build_session_info(file_path: str) -> dict[str, Any] | None:
    """Build SessionInfo from a session file."""
    try:
        if not os.path.exists(file_path):
            return None
        
        entries: list[dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if not entries:
            return None
        
        header = entries[0]
        if header.get("type") != "session":
            return None
        
        stats = os.stat(file_path)
        message_count = 0
        first_message = ""
        all_messages: list[str] = []
        name: str | None = None
        
        for entry in entries:
            if entry.get("type") == "session_info":
                entry_name = entry.get("name")
                if entry_name:
                    name = entry_name.strip()
            
            if entry.get("type") != "message":
                continue
            
            message_count += 1
            msg = entry.get("message", {})
            if not _is_message_with_content(msg):
                continue
            if msg.get("role") not in ("user", "assistant"):
                continue
            
            text_content = _extract_text_content(msg)
            if not text_content:
                continue
            
            all_messages.append(text_content)
            if not first_message and msg.get("role") == "user":
                first_message = text_content
        
        cwd = header.get("cwd", "")
        parent_session_path = header.get("parentSession")
        
        # Use file mtime as modified time
        modified = datetime.fromtimestamp(stats.st_mtime)
        timestamp_str = header.get("timestamp", "")
        if timestamp_str:
            try:
                created = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except Exception:
                created = datetime.fromtimestamp(stats.st_mtime)
        else:
            created = datetime.fromtimestamp(stats.st_mtime)
        
        return {
            "path": file_path,
            "id": header.get("id", ""),
            "cwd": cwd,
            "name": name,
            "parentSessionPath": parent_session_path,
            "created": created,
            "modified": modified,
            "messageCount": message_count,
            "firstMessage": first_message or "(no messages)",
            "allMessagesText": " ".join(all_messages),
        }
    except Exception:
        return None


def _list_sessions_from_dir(dir_path: str) -> list[dict[str, Any]]:
    """List all sessions from a directory."""
    sessions: list[dict[str, Any]] = []
    if not os.path.exists(dir_path):
        return sessions
    
    try:
        files = [f for f in os.listdir(dir_path) if f.endswith(".jsonl")]
        for filename in files:
            file_path = os.path.join(dir_path, filename)
            info = _build_session_info(file_path)
            if info:
                sessions.append(info)
    except Exception:
        pass
    
    return sessions


class SessionHeader:
    def __init__(
        self,
        *,
        session_id: str,
        cwd: str,
        timestamp: str | None = None,
        parent_session: str | None = None,
        version: int = CURRENT_SESSION_VERSION,
    ):
        self.type = "session"
        self.version = version
        self.id = session_id
        self.timestamp = timestamp or datetime.utcnow().isoformat() + "Z"
        self.cwd = cwd
        self.parent_session = parent_session

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "version": self.version,
            "id": self.id,
            "timestamp": self.timestamp,
            "cwd": self.cwd,
            **({"parentSession": self.parent_session} if self.parent_session else {}),
        }


class SessionEntryBase:
    def __init__(self, *, entry_id: str, parent_id: str | None, timestamp: str | None = None):
        self.id = entry_id
        self.parentId = parent_id
        self.timestamp = timestamp or datetime.utcnow().isoformat() + "Z"


class SessionMessageEntry(SessionEntryBase):
    def __init__(self, *, entry_id: str, parent_id: str | None, message: dict, timestamp: str | None = None):
        super().__init__(entry_id=entry_id, parent_id=parent_id, timestamp=timestamp)
        self.type = "message"
        self.message = message

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "id": self.id,
            "parentId": self.parentId,
            "timestamp": self.timestamp,
            "message": self.message,
        }


class ThinkingLevelChangeEntry(SessionEntryBase):
    def __init__(self, *, entry_id: str, parent_id: str | None, thinking_level: str, timestamp: str | None = None):
        super().__init__(entry_id=entry_id, parent_id=parent_id, timestamp=timestamp)
        self.type = "thinking_level_change"
        self.thinkingLevel = thinking_level

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "id": self.id,
            "parentId": self.parentId,
            "timestamp": self.timestamp,
            "thinkingLevel": self.thinkingLevel,
        }


class ModelChangeEntry(SessionEntryBase):
    def __init__(
        self, *, entry_id: str, parent_id: str | None, provider: str, model_id: str, timestamp: str | None = None
    ):
        super().__init__(entry_id=entry_id, parent_id=parent_id, timestamp=timestamp)
        self.type = "model_change"
        self.provider = provider
        self.modelId = model_id

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "id": self.id,
            "parentId": self.parentId,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "modelId": self.modelId,
        }


class CompactionEntry(SessionEntryBase):
    def __init__(
        self,
        *,
        entry_id: str,
        parent_id: str | None,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ):
        super().__init__(entry_id=entry_id, parent_id=parent_id, timestamp=timestamp)
        self.type = "compaction"
        self.summary = summary
        self.firstKeptEntryId = first_kept_entry_id
        self.tokensBefore = tokens_before
        self.details = details

    def to_dict(self) -> dict:
        result = {
            "type": self.type,
            "id": self.id,
            "parentId": self.parentId,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "firstKeptEntryId": self.firstKeptEntryId,
            "tokensBefore": self.tokensBefore,
        }
        if self.details:
            result["details"] = self.details
        return result


def _generate_id(existing_ids: set[str]) -> str:
    for _ in range(100):
        id_str = str(uuid.uuid4())[:8]
        if id_str not in existing_ids:
            return id_str
    return str(uuid.uuid4())


def _get_default_session_dir(cwd: str, agent_dir: str | None = None) -> str:
    if agent_dir is None:
        agent_dir = os.path.expanduser("~/.pi-py")
    # Normalize path separators and remove leading slash
    normalized = cwd.replace("\\", "/").lstrip("/")
    # Replace all path separators and colons with dashes
    safe_path = f"--{normalized.replace('/', '-').replace(':', '-')}--"
    session_dir = os.path.join(agent_dir, "sessions", safe_path)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


class SessionManager:
    def __init__(
        self,
        cwd: str,
        *,
        session_file: str | None = None,
        session_dir: str | None = None,
        agent_dir: str | None = None,
        persist: bool = True,
    ):
        self.cwd = os.path.abspath(cwd)
        self.persist = persist
        self.session_dir = session_dir or _get_default_session_dir(self.cwd, agent_dir)
        self.session_file: str | None = session_file
        self.session_id: str = ""
        self.file_entries: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.leaf_id: str | None = None
        self.flushed = False

        if session_file:
            self.set_session_file(session_file)
        else:
            self.new_session()

    def new_session(self, *, parent_session: str | None = None) -> str | None:
        self.session_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        header = SessionHeader(session_id=self.session_id, cwd=self.cwd, parent_session=parent_session)
        self.file_entries = [header.to_dict()]
        self.by_id.clear()
        self.leaf_id = None
        self.flushed = False

        if self.persist:
            file_timestamp = timestamp.replace(":", "-").replace(".", "-")
            self.session_file = os.path.join(self.session_dir, f"{file_timestamp}_{self.session_id}.jsonl")
            return self.session_file
        return None

    def set_session_file(self, session_file: str) -> None:
        self.session_file = os.path.abspath(session_file)
        if os.path.exists(self.session_file):
            self.file_entries = self._load_entries_from_file(self.session_file)
            if not self.file_entries:
                self.new_session()
                self.session_file = session_file
                self._rewrite_file()
                self.flushed = True
                return
            header = next((e for e in self.file_entries if e.get("type") == "session"), None)
            self.session_id = header.get("id", str(uuid.uuid4())) if header else str(uuid.uuid4())
            self._build_index()
            self.flushed = True
        else:
            explicit_path = self.session_file
            self.new_session()
            self.session_file = explicit_path

    def _load_entries_from_file(self, file_path: str) -> list[dict]:
        if not os.path.exists(file_path):
            return []
        entries = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

    def _build_index(self) -> None:
        self.by_id.clear()
        self.leaf_id = None
        for entry in self.file_entries:
            if entry.get("type") == "session":
                continue
            entry_id = entry.get("id")
            if entry_id:
                self.by_id[entry_id] = entry
                self.leaf_id = entry_id

    def _rewrite_file(self) -> None:
        if not self.persist or not self.session_file:
            return
        with open(self.session_file, "w", encoding="utf-8") as f:
            for entry in self.file_entries:
                f.write(json.dumps(entry) + "\n")

    def _persist(self, entry: dict) -> None:
        if not self.persist or not self.session_file:
            return
        has_assistant = any(
            e.get("type") == "message" and e.get("message", {}).get("role") == "assistant"
            for e in self.file_entries
        )
        if not has_assistant:
            self.flushed = False
            return
        if not self.flushed:
            with open(self.session_file, "w", encoding="utf-8") as f:
                for e in self.file_entries:
                    f.write(json.dumps(e) + "\n")
            self.flushed = True
        else:
            with open(self.session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def _append_entry(self, entry: dict) -> None:
        self.file_entries.append(entry)
        entry_id = entry.get("id")
        if entry_id:
            self.by_id[entry_id] = entry
            self.leaf_id = entry_id
        self._persist(entry)

    def append_message(self, message: dict) -> str:
        existing_ids = set(self.by_id.keys())
        entry_id = _generate_id(existing_ids)
        entry = SessionMessageEntry(entry_id=entry_id, parent_id=self.leaf_id, message=message)
        self._append_entry(entry.to_dict())
        return entry_id

    def append_thinking_level_change(self, thinking_level: str) -> str:
        existing_ids = set(self.by_id.keys())
        entry_id = _generate_id(existing_ids)
        entry = ThinkingLevelChangeEntry(entry_id=entry_id, parent_id=self.leaf_id, thinking_level=thinking_level)
        self._append_entry(entry.to_dict())
        return entry_id

    def append_model_change(self, provider: str, model_id: str) -> str:
        existing_ids = set(self.by_id.keys())
        entry_id = _generate_id(existing_ids)
        entry = ModelChangeEntry(entry_id=entry_id, parent_id=self.leaf_id, provider=provider, model_id=model_id)
        self._append_entry(entry.to_dict())
        return entry_id

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict[str, Any] | None = None,
    ) -> str:
        existing_ids = set(self.by_id.keys())
        entry_id = _generate_id(existing_ids)
        entry = CompactionEntry(
            entry_id=entry_id,
            parent_id=self.leaf_id,
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
        )
        self._append_entry(entry.to_dict())
        return entry_id

    def get_session_id(self) -> str:
        return self.session_id

    def get_session_file(self) -> str | None:
        return self.session_file

    def get_cwd(self) -> str:
        return self.cwd

    def get_session_dir(self) -> str:
        return self.session_dir

    def get_leaf_id(self) -> str | None:
        return self.leaf_id

    def get_leaf_entry(self) -> dict | None:
        if not self.leaf_id:
            return None
        return self.by_id.get(self.leaf_id)

    def get_entry(self, entry_id: str) -> dict | None:
        return self.by_id.get(entry_id)

    def get_branch(self, leaf_id: str | None = None) -> list[dict]:
        target_id = leaf_id or self.leaf_id
        if not target_id:
            return []
        path = []
        current_id: str | None = target_id
        while current_id:
            entry = self.by_id.get(current_id)
            if not entry:
                break
            path.insert(0, entry)
            current_id = entry.get("parentId")
        return path

    def build_session_context(self, leaf_id: str | None = None) -> dict:
        """
        Build session context from entries using tree traversal.
        If leaf_id is provided, walks from that entry to root.
        Handles compaction and branch summaries along the path.
        Returns dict with messages, thinkingLevel, and model.
        """
        # Find leaf entry
        if leaf_id is None:
            leaf_id = self.leaf_id

        if leaf_id is None:
            return {"messages": [], "thinkingLevel": "off", "model": None}

        # Get branch path from leaf to root
        branch = self.get_branch(leaf_id)
        if not branch:
            return {"messages": [], "thinkingLevel": "off", "model": None}

        # Extract settings and find compaction
        thinking_level = "off"
        model: dict[str, str] | None = None
        compaction: dict | None = None

        for entry in branch:
            entry_type = entry.get("type")
            if entry_type == "thinking_level_change":
                thinking_level = entry.get("thinkingLevel", "off")
            elif entry_type == "model_change":
                model = {"provider": entry.get("provider"), "modelId": entry.get("modelId")}
            elif entry_type == "message":
                msg = entry.get("message", {})
                if msg.get("role") == "assistant":
                    # Try to extract model from assistant message
                    if "provider" in msg and "model" in msg:
                        model = {"provider": msg["provider"], "modelId": msg["model"]}
            elif entry_type == "compaction":
                compaction = entry

        # Build messages list
        messages = []

        def append_message(entry: dict) -> None:
            entry_type = entry.get("type")
            if entry_type == "message":
                msg = entry.get("message")
                if msg:
                    messages.append(msg)
            elif entry_type == "custom_message":
                # Create custom message from entry
                content = entry.get("content", "")
                if isinstance(content, str):
                    messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    messages.append({"role": "user", "content": content})
            elif entry_type == "branch_summary":
                summary = entry.get("summary")
                if summary:
                    # Add branch summary as a system-like message
                    messages.append({"role": "user", "content": f"[Branch summary: {summary}]"})

        if compaction:
            # Handle compaction: emit summary first, then kept messages, then messages after compaction
            compaction_summary = compaction.get("summary", "")
            if compaction_summary:
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Previous conversation summarized: {compaction_summary}]",
                    }
                )

            # Find compaction index in branch
            compaction_id = compaction.get("id")
            compaction_idx = next((i for i, e in enumerate(branch) if e.get("id") == compaction_id), -1)

            if compaction_idx >= 0:
                # Emit kept messages (before compaction, starting from firstKeptEntryId)
                first_kept_id = compaction.get("firstKeptEntryId")
                found_first_kept = False
                for i in range(compaction_idx):
                    entry = branch[i]
                    if first_kept_id and entry.get("id") == first_kept_id:
                        found_first_kept = True
                    if found_first_kept or not first_kept_id:
                        append_message(entry)

                # Emit messages after compaction
                for i in range(compaction_idx + 1, len(branch)):
                    append_message(branch[i])
        else:
            # No compaction - emit all messages
            for entry in branch:
                append_message(entry)

        return {"messages": messages, "thinkingLevel": thinking_level, "model": model}

    @staticmethod
    def in_memory(cwd: str = ".") -> "SessionManager":
        return SessionManager(cwd, persist=False)

    @staticmethod
    def create(cwd: str, session_dir: str | None = None, agent_dir: str | None = None) -> "SessionManager":
        return SessionManager(cwd, session_dir=session_dir, agent_dir=agent_dir, persist=True)

    @staticmethod
    def list(cwd: str, session_dir: str | None = None, agent_dir: str | None = None) -> list[dict[str, Any]]:
        """
        List all sessions for a directory.
        Returns list of SessionInfo dicts sorted by modified time (newest first).
        """
        if session_dir is None:
            session_dir = _get_default_session_dir(cwd, agent_dir)
        sessions = _list_sessions_from_dir(session_dir)
        sessions.sort(key=lambda s: s["modified"], reverse=True)
        return sessions

    @staticmethod
    def list_all(agent_dir: str | None = None) -> list[dict[str, Any]]:
        """
        List all sessions across all project directories.
        Returns list of SessionInfo dicts sorted by modified time (newest first).
        """
        sessions_dir = _get_sessions_base_dir(agent_dir)
        if not os.path.exists(sessions_dir):
            return []
        
        sessions: list[dict[str, Any]] = []
        try:
            for item in os.listdir(sessions_dir):
                item_path = os.path.join(sessions_dir, item)
                if os.path.isdir(item_path):
                    sessions.extend(_list_sessions_from_dir(item_path))
        except Exception:
            pass
        
        sessions.sort(key=lambda s: s["modified"], reverse=True)
        return sessions
