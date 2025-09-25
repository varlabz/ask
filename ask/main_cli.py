"""
cli_main.py

CLI entry point for the agent. Run with:
    python src/cli_main.py "Your prompt here"
"""
import argparse
import asyncio
import sys
import contextlib

from ask.core.agent import AgentASK
from ask.core.config import load_config
from ask.core import chat as chat_module
from ask.core.agent_instrumentation import setup_instrumentation

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Run agent.")
    parser.add_argument('-c', '--config', type=str, action='append', help='Path to config yaml (can be used multiple times)')
    parser.add_argument('-s', '--system-prompt', type=str, help='Override system prompt/instructions')
    parser.add_argument('--log', nargs='?', const='ask-log.jsonl', default=None, help='Save agent tracing to a file. Defaults to ask-log.jsonl if no filename is provided.')
    parser.add_argument('-T', '--tchat', action='store_true', help='Start terminal interactive chat mode')
    parser.add_argument("--chat", action="store_true", help="Start chat")
    parser.add_argument("--no-native", action="store_true", help="Start chat without native features")
    parser.add_argument("--chat-port", type=int, help="Explicit chat port (disables auto selection)")
    parser.add_argument('prompt', nargs='*', help='Prompt for the agent')
    args = parser.parse_args()

    # Use default config if none provided
    log_stream_context = open(args.log, "w", encoding="utf-8") if args.log else contextlib.nullcontext()
    with log_stream_context as log_stream:
        if log_stream:
            setup_instrumentation(stream=log_stream)
            
        config = load_config(args.config or [".ask.yaml"])
        if args.system_prompt:
            config.agent.instructions = args.system_prompt

        agent = AgentASK.create_from_config(config)

        # can't use chat and not istty the same time
        if args.tchat and not sys.stdin.isatty():
            print("Error: Interactive chat mode requires a terminal.", file=sys.stderr)
            sys.exit(1)

        # Get prompt from args or stdin
        prompt = ' '.join(args.prompt).strip()
        if not prompt and not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        
        if args.chat:
            from ask.core import chat
            if args.chat_port:
                selected_port = args.chat_port
            else:
                found = chat.find_next_available_port(8000, 9999)
                if found is None:
                    print("No free port available in range 8000-9999", file=sys.stderr)
                    sys.exit(1)
                selected_port = found
                # print(f"Auto-selected port: {selected_port}")

            chat.run_web(agent, selected_port, prompt if prompt else None, native=not args.no_native, reload=False)
            return
        
        if args.tchat:
            from ask.core import tchat
            asyncio.run(agent.run_iter(lambda: tchat.chat(agent, prompt if prompt else None)))
            return
            
        if not prompt:
            print("Error: No prompt provided.", file=sys.stderr)
            parser.print_help(file=sys.stderr)
            sys.exit(1)
        
        result = asyncio.run(agent.run(prompt))
        print(result)

if __name__ in {'__main__', '__mp_main__'}:
    main()
