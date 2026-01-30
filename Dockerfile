FROM public.ecr.aws/neuron/pytorch-training:1.13.1-neuronx-py310

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Disable wandb
ENV WANDB_MODE=disabled

# Neuron cache
ENV NEURON_COMPILE_CACHE_URL=/workspace/neuron_cache
RUN mkdir -p /workspace/neuron_cache

WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip

# ---- Python deps (Neuron-safe, pinned) ----
RUN pip install \
    transformers==4.30.2 \
    tokenizers==0.13.3 \
    peft==0.6.0 \
    accelerate==0.21.0 \
    datasets \
    pyarrow \
    sentencepiece \
    protobuf \
    --index-url https://pypi.org/simple

# ---- Neuron libs ----
RUN pip install \
    neuronx-distributed \
    optimum-neuron \
    --index-url https://pip.repos.neuron.amazonaws.com

# Copy code last
COPY . /workspace
