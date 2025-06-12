#!/bin/bash

forbidden_args=("--model" "--port")

for arg in "$@"; do
    for forbidden in "${forbidden_args[@]}"; do
        if [[ "$arg" == "$forbidden" ]]; then
            echo "Error: Argument '$arg' is not allowed."
            exit 1
        fi
    done
done

fastapi run main.py --port 8000 &

python3 -m vllm.entrypoints.openai.api_server --model $IN_CONTAINER_MODEL_PATH --port 8001 "$@"
