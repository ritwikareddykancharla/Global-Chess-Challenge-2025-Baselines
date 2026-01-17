docker run -it --rm \
  --device=/dev/neuron0 \
  --name neuronx-train-py310 \
  -v $(pwd):/workspace \
  -w /workspace \
  neuronx-training-py310:latest \
  /bin/bash