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
    session_id="eval $(date +%Y-%m-%d_%H-%M-%S) ${model} "
    uv run python -m eval.experiment.main -m "$model" -u "http://bacook.local:12345/v1" -s "$session_id"
    # ssh varis@bacook.local 'ps -ef | grep ollama | awk "/\/Applications/{print \"kill \" \$2}" | sh'
done