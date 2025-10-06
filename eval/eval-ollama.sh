#!/bin/bash
models=(
    "ollama:qwen3:1.7b-q4_K_M"
    "ollama:qwen3:4b-q4_K_M"
    # "ollama:qwen3:8b-q4_K_M"
    # "ollama:qwen3:14b-q4_K_M"
    # "ollama:qwen3:30b-a3b-instruct-2507-q4_K_M"
    # "ollama:qwen3:32b-q4_K_M"
    # "ollama:gpt-oss:20b"
    # "ollama:mistral-small3.2:24b-instruct-2506-q4_K_M"
)

for model in "${models[@]}"
do
    uv run python -m eval.experiment -m "$model"
done