"""
mcp_main.py CLI entry point for MCP client
"""
import argparse
import sys

from config import load_config
from mcp_client import create_mcp_servers

def mcp_main() -> None:
    """Main function for MCP CLI entry point.
    
    Parses arguments, loads config, and initializes MCP clients.
    """
    parser = argparse.ArgumentParser(description="Run MCP client.")
    parser.add_argument('-c', '--config', type=str, default=".ask.yaml", help='Path to ask config yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    mcp_config = getattr(config, 'mcp', None)
    clients = create_mcp_servers(mcp_config)
    print(f"Initialized {len(clients)} MCP client(s).")

if __name__ == "__main__":
    mcp_main()
