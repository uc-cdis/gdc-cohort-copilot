# gdc-cohort-pilot

Run the containerized web app with:
```
docker run -it --rm -p 8000:8000 gdc-cohort-pilot:latest
```

Any additional arguments after the image name are passed to `vllm serve`. The only limitation is that the vLLM model and port arguments should not be overridden.

The app runs on port `8000` within the container, however if port `8000` is occupied on the host, you can remap it: https://docs.docker.com/reference/cli/docker/container/run/#publish

If serving remotely, you may need to ssh tunnel from your local to the remote host:
```
ssh -NL 8000:localhost:8000 <user>@<remote>
```

### TODO
* add instructions about GPU runtime

### Dev Notes
* App server will run on port 8000 and should be exposed external to container
* vLLM server will run on port 8001
* need to make sure outlines cache is placed in container so vllm does not need to rebuild FSM with ever startup
* serve FastAPI app server with:
    ```
    fastapi run main.py --port 8000
    ```
* serve vLLM server with:
    ```
    vllm serve ./model --gpu-memory-utilization 0.95 --port 8001
    ```
* both app and vLLM servers need to be running in parallel
* open localhost:8000 in browser - if serving remotely, ssh tunnel to remote first:
    ```
    ssh -NL 8000:localhost:8000 <user>@<remote>
    ``` 
* One of the steps of our build process for the docker image is to precompute the FSM cache for vllm/outlines. Computing the FSM does not inherently require GPU acceleration, however the trigger to compute the FSM is to generate text using an LLM which does benefit from GPU acceleration. While this step does not strictly require a GPU, a GPU does greatly speed up an already slow process (hence the pre-caching).
    * In order to build the docker image with a GPU runtime is to set the default runtime to "nvidia": https://stackoverflow.com/a/61737404. On the CTDS cluster, this is configured already on node 2.
    * Additionally, you must disable Docker BuildKit by setting the env var to the build command (the legacy builder is deprecated but the GPU at build time hasn't been address in Docker Compose yet: https://github.com/docker/compose/issues/9681):
        ```
        DOCKER_BUILDKIT=0 docker build [...]
        ```