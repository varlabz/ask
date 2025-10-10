#!/bin/bash

models=(
    "lmstudio:qwen/qwen3-1.7b"
    "lmstudio:qwen/qwen3-4b-2507"
    "lmstudio:qwen/qwen3-8b"
    "lmstudio:qwen/qwen3-14b"
    "lmstudio:qwen/qwen3-30b-a3b-2507"
    "lmstudio:qwen/qwen3-32b"
    "lmstudio:openai/gpt-oss-20b"
    "lmstudio:openai-gpt-oss-20b-abliterated-uncensored-neo-imatrix"
    "lmstudio:qwen2.5-1.5b-instruct-mlx"
    "lmstudio:qwen2.5-14b-instruct-mlx"
)

packages=(
    "eval.experiment.simple"
    "eval.experiment.simple_math"
    "eval.experiment.list_of_tools"
    "eval.experiment.review"
    "eval.experiment.list_of_tools_max"
)


for model in "${models[@]}"; do
    session_id="eval $(date +%Y-%m-%d_%H-%M-%S) ${model}"
    uv run python -m eval.experiment.main -m "$model" -s "$session_id" -p "${packages[@]}"
done
