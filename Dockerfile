# =============================================================================
# GE-Lab Training Environment
# Base: PyTorch 2.5.1 + CUDA 12.4 + Python 3.10
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA toolkit (nvcc) for DeepSpeed JIT compilation
RUN conda install -y -c nvidia cuda=12.1 && conda clean -ya

WORKDIR /workspace

# Copy project files
COPY requirements/ requirements/
COPY requirements.txt .
COPY setup.py .
COPY setup.cfg .
COPY MANIFEST.in .
COPY README.md .
COPY swift/ swift/

# Install framework dependencies
RUN pip install --no-cache-dir -r requirements/framework.txt

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Install additional training dependencies
RUN pip install --no-cache-dir \
    deepspeed \
    wandb \
    openai \
    ultralytics \
    easyocr \
    megfile \
    qwen-vl-utils==0.0.11

# Copy rest of the project
COPY . .

# CUDA environment for DeepSpeed
ENV CUDA_HOME=/opt/conda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Runtime environment variables (override at docker run time)
ENV HF_HOME=/data/.cache/huggingface
ENV XDG_CACHE_HOME=/data/.cache
ENV TORCH_HOME=/data/.cache/torch
ENV PYTHONUNBUFFERED=1

# Create cache directories
RUN mkdir -p /workspace/.cache/huggingface \
             /workspace/.cache/torch \
             /workspace/checkpoint \
             /workspace/logs \
             /data

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" && \
    python -c "from swift.llm import sft_main; print('swift OK')"

CMD ["/bin/bash"]
