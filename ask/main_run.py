"""Execute a Python file as a CLI utility.

Usage:
    ask-run <file_path> [-- <script_args...>]
"""

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    """
    Main entry point for the ask-run command.
    """
    parser = argparse.ArgumentParser(
        description="Execute a Python file.",
        usage="ask-run <file_path> [-- <script_args...>]",
    )
    parser.add_argument("file_path", help="The path to the Python file to execute.")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the script. Use '--' to separate script args.",
    )

    args = parser.parse_args()

    script_path = Path(args.file_path).expanduser()
    if not script_path.is_file():
        print(f"Error: File not found '{args.file_path}'", file=sys.stderr)
        sys.exit(1)

    # Prepare sys.argv for the script to be executed
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    
    original_argv = sys.argv
    sys.argv = [str(script_path.resolve())] + script_args

    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"Error executing script: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()


