"""
Interactive mode for pi coding agent - simple REPL interface.
"""

import os
import sys
from datetime import datetime
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import clear

from .session import AgentSession, create_agent_session


class InteractiveMode:
    """Simple interactive REPL for the coding agent."""

    def __init__(self, session: AgentSession):
        self.session = session
        history_file = os.path.expanduser("~/.pi-py/history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        self.prompt_session = PromptSession(history=FileHistory(history_file))
        self.running = True

    def print_message(self, role: str, content: str) -> None:
        """Print a message with role indicator."""
        if role == "user":
            print(f"\n👤 You: {content}")
        elif role == "assistant":
            print(f"\n🤖 Assistant: {content}")
        else:
            print(f"\n{role}: {content}")

    def print_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Print tool execution info."""
        args_str = ", ".join(f"{k}={v[:50]}..." if len(str(v)) > 50 else f"{k}={v}" for k, v in arguments.items())
        print(f"  🔧 {tool_name}({args_str})")

    def run(self) -> None:
        """Run the interactive loop."""
        print("Pi Coding Agent (Python) - Interactive Mode")
        print("Type '/quit' or '/exit' to exit, '/clear' to clear screen, '/help' for help\n")

        # Subscribe to tool calls
        def on_tool_call(tool_call_id: str, tool_name: str, arguments: dict) -> None:
            self.print_tool_call(tool_name, arguments)

        self.session.subscribe_tool_call(on_tool_call)

        while self.running:
            try:
                # Get user input
                text = self.prompt_session.prompt("pi-py> ").strip()

                if not text:
                    continue

                # Handle commands
                if text.startswith("/"):
                    self.handle_command(text)
                    continue

                # Send message to agent
                print("\n⏳ Processing...")
                try:
                    response = self.session.prompt(text)
                    self.print_message("assistant", response)
                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted")
                except Exception as e:
                    print(f"\n❌ Error: {e}")

            except KeyboardInterrupt:
                print("\n\nUse '/quit' to exit or Ctrl+C again to force quit")
                try:
                    text = self.prompt_session.prompt("pi-py> ").strip()
                    if text.lower() in ("/quit", "/exit"):
                        break
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
            except EOFError:
                print("\n\nExiting...")
                break

    def handle_command(self, cmd: str) -> None:
        """Handle slash commands."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ("/quit", "/exit", "/q"):
            self.running = False
            print("Goodbye!")
        elif command in ("/clear", "/c"):
            clear()
        elif command in ("/help", "/h"):
            self.print_help()
        elif command == "/model":
            if args:
                self.session.set_model(args)
                print(f"Model set to: {args}")
            else:
                print(f"Current model: {self.session.model}")
        elif command == "/session":
            session_file = self.session.session_manager.get_session_file()
            if session_file:
                print(f"Session file: {session_file}")
                print(f"Session ID: {self.session.session_manager.get_session_id()}")
            else:
                print("In-memory session (not persisted)")
        elif command == "/cwd":
            print(f"Working directory: {self.session.cwd}")
        elif command == "/compact" or command.startswith("/compact "):
            custom_instructions = args if command.startswith("/compact ") else None
            self.handle_compact_command(custom_instructions)
        elif command == "/resume":
            self.handle_resume_command()
        else:
            print(f"Unknown command: {command}. Type '/help' for available commands.")

    def handle_compact_command(self, custom_instructions: str | None = None) -> None:
        """Handle /compact command - manually trigger compaction."""
        try:
            print("\n⏳ Compacting context...")
            result = self.session.compact(custom_instructions)
            print(f"\n✅ Compaction completed!")
            print(f"   Summary: {result['summary'][:200]}..." if len(result['summary']) > 200 else f"   Summary: {result['summary']}")
            print(f"   Tokens before: {result['tokensBefore']}")
            print(f"   First kept entry ID: {result['firstKeptEntryId']}")
        except ValueError as e:
            print(f"\n⚠️  {e}")
        except Exception as e:
            print(f"\n❌ Compaction failed: {e}")

    def handle_resume_command(self) -> None:
        """Handle /resume command - show session selector and resume a session."""
        from .session_manager import SessionManager
        
        print("\n📋 Loading sessions...")
        sessions = SessionManager.list_all()
        
        if not sessions:
            print("No sessions found.")
            return
        
        print(f"\nFound {len(sessions)} session(s):\n")
        for i, session in enumerate(sessions, 1):
            name = session.get("name") or "(unnamed)"
            cwd = session.get("cwd", "")
            msg_count = session.get("messageCount", 0)
            first_msg = session.get("firstMessage", "")[:60]
            modified = session.get("modified")
            if isinstance(modified, datetime):
                mod_str = modified.strftime("%Y-%m-%d %H:%M")
            else:
                mod_str = str(modified)
            
            print(f"  {i}. {name}")
            print(f"     CWD: {cwd}")
            print(f"     Messages: {msg_count}, Modified: {mod_str}")
            if first_msg:
                print(f"     First: {first_msg}...")
            print()
        
        try:
            choice = input("Select session number (or Enter to cancel): ").strip()
            if not choice:
                print("Cancelled.")
                return
            
            idx = int(choice) - 1
            if idx < 0 or idx >= len(sessions):
                print(f"Invalid selection: {choice}")
                return
            
            selected = sessions[idx]
            session_path = selected["path"]
            
            print(f"\n⏳ Resuming session: {selected.get('name') or '(unnamed)'}...")
            self.session.switch_session(session_path)
            print(f"✅ Session resumed!")
            print(f"   Session file: {session_path}")
            print(f"   Messages: {selected.get('messageCount', 0)}")
        except ValueError:
            print(f"Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
        except Exception as e:
            print(f"\n❌ Failed to resume session: {e}")

    def print_help(self) -> None:
        """Print help message."""
        print("\nAvailable commands:")
        print("  /quit, /exit, /q    - Exit the interactive mode")
        print("  /clear, /c          - Clear the screen")
        print("  /help, /h           - Show this help message")
        print("  /model [name]       - Show or set the model")
        print("  /session            - Show session info")
        print("  /cwd                - Show current working directory")
        print("  /compact [instructions] - Manually compact context (optional custom instructions)")
        print("  /resume             - Resume a previous session")
        print("\nType your message and press Enter to send it to the agent.")


def run_interactive(
    *,
    cwd: str | None = None,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    """Run interactive mode."""
    session = create_agent_session(
        cwd=cwd or os.getcwd(),
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    mode = InteractiveMode(session)
    mode.run()
