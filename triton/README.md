# Triton
Hơw to run: 
- Add torchscript model to model_repository
- Run Triton: docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/triton/model_repository:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models

## Prerequisites

```
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker

nvidia-container-cli --version
```