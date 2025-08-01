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

def cli_main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Run agent.")
    parser.add_argument('-c', '--config', type=str, default=".ask.yaml", help='Path to ask config yaml')
    parser.add_argument('-s', '--system-prompt', type=str, default=None, help='Overwrite system prompt/instructions')
    parser.add_argument('prompt', nargs='*', help='Prompt for the agent')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.system_prompt: config.agent.instructions = args.system_prompt
    
    agent = AgentASK.create(config)
    prompt_str = ' '.join(args.prompt).strip()
    if not prompt_str:
        if not sys.stdin.isatty():
            prompt_str = sys.stdin.read().strip()
        else:
            raise ValueError("No prompt provided. Please provide a prompt as an argument or via stdin.")

    if not prompt_str:
        print("Error: No prompt provided.", file=sys.stderr)
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(agent.run(prompt_str))
    print(result)

if __name__ == '__main__':
    cli_main()
