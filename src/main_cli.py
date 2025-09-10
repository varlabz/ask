"""
cli_main.py

CLI entry point for the agent. Run with:
    python src/cli_main.py "Your prompt here"
"""
import argparse
import asyncio
import sys

from core.agent import AgentASK
from core.config import load_config
from core.chat import chat

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Run agent.")
    parser.add_argument('-c', '--config', type=str, action='append', help='Path to config yaml (can be used multiple times)')
    parser.add_argument('-s', '--system-prompt', type=str, help='Override system prompt/instructions')
    parser.add_argument('--chat', action='store_true', help='Enter interactive chat mode')
    parser.add_argument("--web", action="store_true", help="Start web UI")
    parser.add_argument("--web-port", type=int, default=8004, help="port")
    parser.add_argument('prompt', nargs='*', help='Prompt for the agent')
    args = parser.parse_args()

    # Use default config if none provided
    config = load_config(args.config or [".ask.yaml"])
    if args.system_prompt:
        config.agent.instructions = args.system_prompt

    agent = AgentASK.create_from_config(config)

    # can't use chat and not istty the same time
    if args.chat and not sys.stdin.isatty():
        print("Error: Interactive chat mode requires a terminal.", file=sys.stderr)
        sys.exit(1)

    # Get prompt from args or stdin
    prompt = ' '.join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    
    if args.web:
        import core.web
        core.web.run_web(agent, args.web_port, prompt if prompt else None)
    elif args.chat:
        asyncio.run(agent.run_iter(lambda: chat(agent, prompt if prompt else None)))
        sys.exit(0)
    elif not prompt:
        print("Error: No prompt provided.", file=sys.stderr)
        parser.print_help(file=sys.stderr)
        sys.exit(1)
    else:
        result = asyncio.run(agent.run(prompt))
        print(result)

if __name__ in {'__main__', '__mp_main__'}:
    main()
