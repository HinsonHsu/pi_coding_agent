"""
CLI for pi coding agent. Print mode: pi-py -p "prompt" outputs the response and exits.
Interactive mode: pi-py (no flags) starts an interactive REPL.
"""

import argparse
import os
import sys

from .interactive import run_interactive
from .session import create_agent_session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pi coding agent (Python). Default: interactive mode. Use -p for print mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pi-py                    # Start interactive mode
  pi-py -p "List files"    # Print mode: send prompt and exit
  pi-py --model gpt-4o-mini -p "Hello"
        """,
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print_mode",
        action="store_true",
        help="Print mode: send prompt, output response text, exit",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PI_MODEL", "gpt-4o"),
        help="Model ID (default: gpt-4o or PI_MODEL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="API base URL for OpenAI-compatible endpoint (default: OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="Prompt to send (joined as one message, only used in print mode)",
    )
    args = parser.parse_args()

    # Print mode
    if args.print_mode:
        message = " ".join(args.message).strip()
        if not message:
            print("Error: provide a message in print mode, e.g. pi-py -p 'List files in this directory'", file=sys.stderr)
            sys.exit(1)

        try:
            session = create_agent_session(
                cwd=os.getcwd(),
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
            )
            response = session.prompt(message)
            print(response)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Interactive mode (default)
        try:
            run_interactive(
                cwd=os.getcwd(),
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
            )
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
