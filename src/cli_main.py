"""
cli_main.py

CLI entry point for the agent. Run with:
    python src/cli_main.py "Your prompt here"
"""
import argparse
import asyncio
import sys

from agent import AgentASK
from config import load_config
from chat import chat


def cli_main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Run agent.")
    parser.add_argument('-c', '--config', type=str, action='append', help='Path to config yaml (can be used multiple times)')
    parser.add_argument('-s', '--system-prompt', type=str, help='Override system prompt/instructions')
    parser.add_argument('--chat', action='store_true', help='Enter interactive chat mode')
    parser.add_argument('prompt', nargs='*', help='Prompt for the agent')
    args = parser.parse_args()

    # Use default config if none provided
    config = load_config(args.config or [".ask.yaml"])
    if args.system_prompt:
        config.agent.instructions = args.system_prompt

    agent = AgentASK.create_from_config(config)

    # Get prompt from args or stdin
    prompt_str = ' '.join(args.prompt).strip()
    if not prompt_str and not sys.stdin.isatty():
        prompt_str = sys.stdin.read().strip()
    
    if args.chat:
        asyncio.run(agent.run(lambda: chat(agent, prompt_str if prompt_str else None)))
        sys.exit(0)

    if not prompt_str:
        print("Error: No prompt provided.", file=sys.stderr)
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(agent.run(agent.iter(prompt_str)))
    print(result)

if __name__ == '__main__':
    cli_main()
