#!/bin/bash

fastapi run main.py --port 8000 &

python3 -m vllm.entrypoints.openai.api_server --model $IN_CONTAINER_MODEL_PATH --port 8001 "$@"
