# Dockerfile for Trainium training - uses newer SDK with NeuronSFTTrainer
FROM public.ecr.aws/neuron/pytorch-training-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04

WORKDIR /workspace

# Install compatible versions of required packages
RUN pip install --no-cache-dir \
    'datasets==2.14.0' \
    'pyarrow==14.0.0' \
    'optimum-neuron>=0.0.25' \
    'peft>=0.7.0' \
    'trl>=0.7.0' \
    'fsspec==2024.2.0'
