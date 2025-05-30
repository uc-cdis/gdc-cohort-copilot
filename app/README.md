
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
    * the caching is fragile and requires the model path specified at caching precisely match the model path at inference
    * see Dockerfile for more details
* As one of the steps of our build process for the docker image is to precompute the FSM cache for vllm/outlines. Computing the FSM does not inherently require GPU acceleration, however the trigger to compute the FSM is to generate text using an LLM which does benefit from GPU acceleration. While this step does not strictly require a GPU, a GPU does greatly speed up an already slow process (hence the pre-caching).
    * In order to build the docker image with a GPU runtime is to set the default runtime to "nvidia": https://stackoverflow.com/a/61737404. On the CTDS cluster, this is configured already on node 2.
    * Additionally, you must disable Docker BuildKit by setting the env var to the build command (the legacy builder is deprecated but the GPU at build time hasn't been address in Docker Compose yet: https://github.com/docker/compose/issues/9681):
        ```bash
        DOCKER_BUILDKIT=0 docker build [...]
        ```