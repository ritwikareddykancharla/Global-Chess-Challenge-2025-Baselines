FROM public.ecr.aws/neuron/pytorch-training:2.1.0-neuronx-py310

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Disable wandb
ENV WANDB_MODE=disabled

# Neuron cache
ENV NEURON_COMPILE_CACHE_URL=/workspace/neuron_cache
RUN mkdir -p /workspace/neuron_cache

WORKDIR /workspace

# ---- Python deps (PINNED, COMPATIBLE) ----
RUN pip install --upgrade pip

RUN pip install \
    transformers==4.35.2 \
    tokenizers==0.18.1 \
    peft==0.7.1 \
    datasets \
    pyarrow \
    accelerate \
    --index-url https://pypi.org/simple

# Install Neuron libs WITHOUT deps (critical)
RUN pip install \
    optimum-neuron \
    neuronx-distributed \
    --no-deps \
    --index-url https://pip.repos.neuron.amazonaws.com

# Copy code last (so rebuilds are fast)
COPY . /workspace
