#!/bin/bash
models=(
    "lmstudio:qwen/qwen3-1.7b"
    "lmstudio:qwen/qwen3-4b-2507"
    "lmstudio:qwen/qwen3-8b"
    # "lmstudio:qwen/qwen3-14b"
    # "lmstudio:qwen/qwen3-30b-a3b-2507"
    # "lmstudio:qwen/qwen3-32b"
    # "lmstudio:gpt-oss-20b"
    # "lmstudio:mistral-small3.2:24b-instruct-2506-q4_K_M"
)

for model in "${models[@]}"
do
    uv run python -m eval.experiment -m "$model" -u "http://bacook.local:12345/v1"
done