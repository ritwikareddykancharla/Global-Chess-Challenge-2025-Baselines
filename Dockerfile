FROM public.ecr.aws/neuron/pytorch-training:2.0.1-neuronx-py310

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

# ---- Python deps (PINNED & NEURON-SAFE) ----
RUN pip install \
    transformers==4.35.2 \
    tokenizers==0.14.1 \
    peft==0.7.1 \
    accelerate==0.24.1 \
    datasets \
    pyarrow \
    sentencepiece \
    protobuf \
    --index-url https://pypi.org/simple

# ---- Neuron libs (installed last) ----
RUN pip install \
    optimum-neuron==0.0.19 \
    neuronx-distributed \
    --no-deps \
    --index-url https://pip.repos.neuron.amazonaws.com

# Copy code last
COPY . /workspace
