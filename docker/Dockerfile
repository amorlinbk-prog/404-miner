# Trellis based generator
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG GITHUB_USER
ENV GITHUB_USER=${GITHUB_USER}

ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

ENV DEBIAN_FRONTEND=noninteractive

LABEL name="trellis-gen" maintainer="404gen"

# Install docker dependencies
# Installing python 3.11, enbaling channel
RUN apt update -y &&  \
    apt-get install software-properties-common -y &&  \
    apt update -y &&  \
    add-apt-repository ppa:deadsnakes/ppa -y &&  \
    apt update -y  &&\
    apt install -y wget gnupg

# Add NVIDIA repo for CUDA 12.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    ninja-build \
    python3.11  \
    python3-pip  \
    python3-wheel \
    libjpeg-dev  \
    zlib1g-dev \
    cuda-toolkit-12-8 \
    cuda-nvcc-12-8 \
    cuda-libraries-dev-12-8 \
    cuda-nvtx-12-8 \
    && rm -rf /var/lib/apt/lists/*

# rebinding names for python, pip, gcc, g++
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 120 &&  \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 120

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

# preparing project
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

RUN pip install packaging
RUN pip install flash-attn==2.8.0.post2 --no-build-isolation --no-cache-dir
RUN pip install flashinfer-python==0.5.2 flashinfer-cubin==0.5.2 --no-build-isolation --no-cache-dir
RUN pip install -r requirements.txt

# Copy source code after dependencies
COPY . /app


EXPOSE 10006
CMD ["python",  "serve.py"]