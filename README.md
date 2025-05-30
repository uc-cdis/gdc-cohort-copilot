# GDC Co(hort)-Pilot

Run the containerized web app using the command below before opening http://localhost:8000 in a web browser:
```bash
docker run -it --rm -p 8000:8000 gdc-cohort-pilot:latest
```

* Any additional arguments after the image name are passed to `vllm serve`. The only limitation is that the vLLM model and port arguments should not be overridden.
* vLLM automatically detects and utilizes GPUs if available, however GPUs must first be exposed to the docker container. To enable GPU acceleration within a docker container, follow steps to [install nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and then run the container with the below additional flags:
    ```bash
    docker run -it --rm -p 8000:8000 --runtime nvidia --gpus all gdc-cohort-pilot:latest
    ```
    * Please refer to the `docker run` documentation on [accessing GPUs](https://docs.docker.com/reference/cli/docker/container/run/#gpus) for additional details.
* The app runs on port `8000` within the container, however if port `8000` is occupied on the host, you can remap it. Refer to the `docker run` documentation on [host/container port mapping](https://docs.docker.com/reference/cli/docker/container/run/#publish) for details.
* If serving remotely, you will need to ssh tunnel from your local to the remote host before being able to open the app in your web browser:
    ```bash
    ssh -NL 8000:localhost:8000 <user>@<remote>
    ```
