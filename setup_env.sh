#!/bin/bash

# Stop the script on any error
set -e

# Check for Conda installation and initialize Conda in script
if [ -z "$(which conda)" ]; then
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init --all
    apt update
    apt install nano vim
    apt install npm -y
    npm install -g pm2@6.0.12
else
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# Attempt to find Conda's base directory and source it (required for `conda activate`)
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment and activate it
conda env create -f conda_env.yml
conda activate 404-base-miner
conda info --env

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda.sh" <<'SH'
export CUDA_ENV_PREFIX="$CONDA_PREFIX"
# Save originals if not saved
if [ -z "${CUDA_SAVED_CUDA_HOME+x}" ]; then export CUDA_SAVED_CUDA_HOME="${CUDA_HOME:-}"; fi
if [ -z "${CUDA_SAVED_CUDA_PATH+x}" ]; then export CUDA_SAVED_CUDA_PATH="${CUDA_PATH:-}"; fi
if [ -z "${CUDA_SAVED_LD_LIBRARY_PATH+x}" ]; then export CUDA_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"; fi

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
SH

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda.sh" <<'SH'
if [ -n "${CUDA_SAVED_CUDA_HOME+x}" ]; then export CUDA_HOME="${CUDA_SAVED_CUDA_HOME}"; else unset CUDA_HOME; fi
if [ -n "${CUDA_SAVED_CUDA_PATH+x}" ]; then export CUDA_PATH="${CUDA_SAVED_CUDA_PATH}"; else unset CUDA_PATH; fi
if [ -n "${CUDA_SAVED_LD_LIBRARY_PATH+x}" ]; then export LD_LIBRARY_PATH="${CUDA_SAVED_LD_LIBRARY_PATH}"; fi

if [ -n "$CUDA_ENV_PREFIX" ]; then
    PATH=":${PATH:-}:"; PATH="${PATH//:$CUDA_ENV_PREFIX\/bin:/:}"; PATH="${PATH#:}"; PATH="${PATH%:}"; export PATH
    if [ -z "${CUDA_SAVED_LD_LIBRARY_PATH+x}" ]; then
        LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"; LD_LIBRARY_PATH="${LD_LIBRARY_PATH//:$CUDA_ENV_PREFIX\/lib:/:}"; LD_LIBRARY_PATH="${LD_LIBRARY_PATH#:}"; LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"; export LD_LIBRARY_PATH
    fi
fi
unset CUDA_ENV_PREFIX CUDA_SAVED_CUDA_HOME CUDA_SAVED_CUDA_PATH CUDA_SAVED_LD_LIBRARY_PATH
SH

export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
pip install -r requirements.txt
pip install flash-attn==2.8.0.post2 --no-build-isolation --no-cache-dir
pip install flashinfer-python==0.5.2 flashinfer-cubin==0.5.2 --no-build-isolation --no-cache-dir

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the generation.config.js file for PM2 with specified configurations
cat <<EOF > generation.config.js
module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--port 10006'
  }]
};
EOF

echo -e "\n\n[INFO] generation.config.js generated for PM2."