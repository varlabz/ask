#!/bin/bash

models=(
    # ollama:qwen3:32b-q8_0
    # ollama:qwen3:30b-a3b-instruct-2507-q8_0         
    # ollama:qwen3:14b-q8_0
    # ollama:qwen3:8b-q8_0
    # ollama:qwen3:1.7b-q8_0 

    "ollama:mistral-small3.2:24b-instruct-2506-q4_K_M"
    "ollama:hf.co/ibm-granite/granite-3.3-8b-instruct-GGUF:Q4_K_M"
    "ollama:gpt-oss:20b"
    # "ollama:qwen3:32b-q4_K_M"
    "ollama:qwen3:30b-a3b-instruct-2507-q4_K_M"
    "ollama:qwen3:14b-q4_K_M"
    "ollama:qwen3:8b-q4_K_M"
    "ollama:qwen3:4b-q4_K_M"
    "ollama:qwen3:1.7b-q4_K_M"
)

packages=(
    "eval.experiment.simple"
    "eval.experiment.simple_math"
    "eval.experiment.list_of_tools"
    "eval.experiment.review"
)

for model in "${models[@]}"
do
    session_id="eval $(date +%Y-%m-%d_%H-%M-%S) ${model} "
    uv run --active python -m eval.experiment.main -m $model -u http://bacook.local:11434/v1  -s "$session_id" -p "${packages[@]}"
    # ssh varis@bacook.local 'ps -ef | grep ollama | awk "/\/Applications/{print \"kill \" \$2}" | sh'
done