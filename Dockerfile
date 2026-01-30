# Dockerfile for Trainium training
FROM public.ecr.aws/neuron/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.18.2-ubuntu20.04

WORKDIR /workspace

# Install compatible versions of required packages
RUN pip install --no-cache-dir \
    'datasets==2.14.0' \
    'wandb' \
    'optimum-neuron==0.0.21' \
    'peft==0.7.0' \
    'pyarrow' \
    'fsspec==2024.2.0'

# Copy requirements if needed
# COPY requirements.txt .
# RUN pip install -r requirements.txt
