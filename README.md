
______________________________________________________________________

<div align="center">

# 3D Medical Image Segmentation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This repository provides a comprehensive approach to 3D medical image segmentation, covering both model development and deployment. It includes:

### Model Training:
- Training models from scratch.
- Fine-tuning on existing models.
- Leveraging foundation models to boost segmentation performance.
 
### Loss Functions:
- Implementation and evaluation of multiple loss functions specifically designed for medical image segmentation.

### Inference and Deployment:
- Inference pipeline designed for clinical usability.
- Integration with 3D Slicer and OHIF Viewer.
- Serving support via MONAI Label and Triton Inference Server for inference and visualization.
- Docker for containerization

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/minhminh2k/3D-SegClass-Medical
cd 3D-SegClass-Medical

# [OPTIONAL] create conda environment
conda create -n mis python=3.9
conda activate mis

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/minhminh2k/3D-SegClass-Medical
cd 3D-SegClass-Medical

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=1

# Deploy with Docker and Triton Inference Server


## Experiment results
- Results and Visualization: [!Wandb](https://wandb.ai/minhqd9112003/3d-segmentation)
```
