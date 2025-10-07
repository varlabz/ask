#!/bin/bash

models=(
    "ollama:mistral-small3.2:24b-instruct-2506-q4_K_M"
    "ollama:hf.co/ibm-granite/granite-3.3-8b-instruct-GGUF:Q4_K_M"
    "ollama:gpt-oss:20b"
    "ollama:qwen3:32b-q4_K_M"
    "ollama:qwen3:30b-a3b-instruct-2507-q4_K_M"
    "ollama:qwen3:14b-q4_K_M"
    "ollama:qwen3:8b-q4_K_M"
    "ollama:qwen3:4b-q4_K_M"
    "ollama:qwen3:1.7b-q4_K_M"
)

session_id="eval ${#models[@]} models"

for model in "${models[@]}"
do
    uv run --active python -m eval.experiment.one_shoot_agent -m $model -u http://bacook.local:11434/v1  -s "$session_id"
    ssh varis@bacook.local 'ps -ef | grep ollama | awk "/\/Applications/{print \"kill \" \$2}" | sh'
done