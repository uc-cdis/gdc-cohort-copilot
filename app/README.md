
### Dev Notes

* App server runs on port `8000` inside the container and should be exposed externally
* vLLM server runs on port `8001` inside the container
* Both app and vLLM servers need to be running in parallel
    * serve FastAPI app server with:
        ```bash
        fastapi run main.py --port 8000
        ```bash
    * serve vLLM server with:
        ```
        vllm serve --model ../model --port 8001
        ```
        * the model path passed to `serve` must precisely match the model path used for FSM caching
* vLLM is configured to use outlines for guided decoding of the cohort JSON
    * computing the FSM takes time and so should be be precomputed and cached
    * outlines caching currently does not play nice with vLLM but can be easily patched/hacked
    * the caching is fragile for 2 reasons:
        * it requires the model path specified at caching precisely match the model path at inference
        * the helper file (`cache-fsm.py`) uses some internal vLLM tools that are not part of the official API and are thus liable to change
    * these details are handled by the Dockerfile
        * the docker build process does NOT require a GPU
