# gdc-cohort-pilot


### Dev Notes
* App server will run on port 8000 and should be exposed external to container
* vLLM server will run on port 8001
* need to make sure outlines cache is placed in container so vllm does not need to rebuild FSM with ever startup
* serve FastAPI server with:
    ```
    fastapi run main.py --port 8000
    ```
* serve vLLM server with:
    ```
    vllm serve ./model --gpu-memory-utilization 0.95 --port 8001
    ```
