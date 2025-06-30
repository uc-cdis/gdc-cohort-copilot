# Containerization

Build the docker image **from the repo root** using the following command (don't forget the trailing `.`):
```bash
docker build \
--tag quay.io/cdis/gdc-cohort-copilot:latest \
--build-context model=/path/to/model \
--file docker/Dockerfile .
```
