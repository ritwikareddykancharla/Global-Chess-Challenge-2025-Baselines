#!/bin/bash
# Use official AWS Neuron Docker image
docker run -it --rm \
  --device=/dev/neuron0 \
  --name neuronx-training \
  -v $(pwd):/workspace \
  -w /workspace \
  public.ecr.aws/neuron/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.18.2-ubuntu20.04 \
  /bin/bash