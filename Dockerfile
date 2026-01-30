# Dockerfile for Trainium training
# KEY: Only add datasets, do NOT upgrade any pre-installed packages
FROM public.ecr.aws/neuron/pytorch-training-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04

WORKDIR /workspace

# ONLY install datasets - do NOT install or upgrade optimum-neuron, peft, etc.
# The base image has pre-tested compatible versions
RUN pip install --no-cache-dir --no-deps \
    'datasets==2.14.0' \
    'pyarrow==14.0.0' \
    'xxhash' \
    'multiprocess' \
    'dill' \
    'fsspec'
