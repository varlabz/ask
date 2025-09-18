FROM debian:stable-slim

RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN npm install -g npm

RUN apt-get update && \
    apt-get install -y python3 git ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh    

ENV PATH="/app:/root/.local/bin:$PATH"

ENV UV_CACHE_DIR=/tmp/.uv-cache
ENV npm_config_cache=/tmp/.npm-cache

WORKDIR /app
COPY ./examples /app

ENTRYPOINT []
CMD ["bash"]

LABEL description="Docker image for ASK CLI/MCP - A command-line interface and MCP server"
LABEL version="0.1.0"
LABEL repository="https://github.com/varlabz/ask"
LABEL usage.example="docker run --rm -it ask uvx --from git+https://github.com/varlabz/ask ask-cli --help"
LABEL usage.with_config="docker run --rm -it -v \$HOME/.config/ask:/root/.config/ask:ro -v ./:/app -v <docker-volume>:/tmp --network=host ask <args>"
