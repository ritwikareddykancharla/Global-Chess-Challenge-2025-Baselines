#!/bin/bash
# Use custom Docker image with all dependencies
docker run -it --rm \
  --device=/dev/neuron0 \
  --name neuronx-training \
  -v $(pwd):/workspace \
  -w /workspace \
  neuronx-training-py310:latest \
  /bin/bash