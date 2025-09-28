import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

common_mcp = {
    # Search via searxng-mcp (update SEARX_HOST if needed)
    "search": {
        "command": [
            "uvx",
            "--from",
            "git+https://github.com/varlabz/searxng-mcp",
            "mcp-server",
        ],
        "env": {"SEARX_HOST": "http://macook.local:8080"},
    },
    # Generic fetcher for web content
    "fetch": {"command": ["uvx", "mcp-server-fetch", "--ignore-robots-txt"]},
    # YouTube content summarizer/extractor
    "youtube": {
        "command": ["npx", "-y", "https://github.com/varlabz/youtube-mcp", "--mcp"]
    },
    # Convert HTML -> Markdown (and other formats)
    "converter": {
        "command": [
            "uvx",
            "--from",
            "git+https://github.com/varlabz/markitdown-mhtml.git@mhtml#subdirectory=packages/markitdown-mcp",
            "markitdown-mcp",
        ],
        "env": {"MARKITDOWN_ENABLE_PLUGINS": "true"},
    },
    # Optional: filesystem access (prefix fs:)
    "filesystem": {
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."],
        "tool_prefix": "fs",
    },
    # In-memory store for gathered markdown contexts (prefix mem:)
    "memory": {"command": ["uvx", "basic-memory", "mcp"], "tool_prefix": "mem"},
    # Chain-of-thought planning without logging thoughts
    "sequential_thinking": {
        "command": ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
        "env": {"DISABLE_THOUGHT_LOGGING": "true"},
    },
}
