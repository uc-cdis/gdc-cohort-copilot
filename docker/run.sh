#!/bin/bash

# TODO start app


python3 -m vllm.entrypoints.openai.api_server "$@"
