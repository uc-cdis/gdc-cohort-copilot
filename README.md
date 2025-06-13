# GDC Co(hort)-Pilot

We recommend using `docker` to run **GDC Co(hort)-Pilot**. Run the command below before opening http://localhost:8000 in a web browser:
```bash
docker run -it --rm -p 8000:8000 --runtime nvidia --gpus all quay.io/cdis/gdc-cohort-pilot:latest
```

* Our image requires GPU acceleration to run:
    * Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if not already installed.
    * Please refer to the `docker run` documentation on [accessing GPUs](https://docs.docker.com/reference/cli/docker/container/run/#gpus) for additional details.
* Any additional arguments after the image name are passed to `vllm serve`. The only limitation is that the vLLM model and port arguments should not be overridden.
* The app runs on port `8000` within the container, however if port `8000` is occupied on the host, you can remap it. Refer to the `docker run` documentation on [host/container port mapping](https://docs.docker.com/reference/cli/docker/container/run/#publish) for details.
* If serving remotely, you will need to ssh tunnel from your local to the remote host before being able to open the app in your web browser:
    ```bash
    ssh -NL 8000:localhost:8000 <user>@<remote>
    ```

## Cohort-LLM

In addition to the containerized application, we also include our source code for developing and evaluating **Cohort-LLM**, the generative language model powering the GDC-Cohort-Pilot. In order, the steps for our experiments are:
1. Setup and activate development environment
    ```
    conda env create -f env.yaml
    conda activate cohort
    ```
1. [Data Preprocessing](./data-preprocessing)
1. [Synthetic Data Generation](./data-generation)
1. [Model Training and Inference](./cohort-llm)
1. [OpenAI Comparison](./openai-prompting)
1. [Evaluation](./evaluation)
1. [Containerization](./docker)

## Citation

```
@article{song2025gdc,
  title={GDC-Cohort-Pilot: An AI Copilot for Curating Cohorts from the Genomic Data Commons},
  author={Song, Steven and Subramanyam, Anirudh and Zhang, Zhenyu and Venkat, Aarti and Grossman, Robert L},
  journal={TODO},
  year={2025}
}
```

## TODO

* Remove `cases.` from `files.` properties:
    * Regenerate synthetic data
    * Retrain models
    * Re-evaluate models
    * Update paper
    * Remove hotfix from app
    * Update docker image
