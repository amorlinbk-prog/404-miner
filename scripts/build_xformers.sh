# Attempt to find Conda's base directory and source it (required for `conda activate`)
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment and activate it
conda activate 404-base-miner
conda install nvidia::libcublas-dev
conda install nvidia::libcusparse-dev
conda install nvidia::libcusolver-dev

CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
CUDA_HOME=$CONDA_PREFIX
PATH=$CONDA_PREFIX/bin:$PATH
PATH="$CUDA_HOME/bin:$PATH"
CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include"
LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib"
CONDA_INTERPRETER_PATH=$(which python)

export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=16
export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

git clone https://github.com/facebookresearch/xformers.git --recursive
cd xformers

mkdir /workspace/compiled_wheels

pip install build wheel setuptools ninja psutil packaging einops
python -m build --wheel --no-isolation --outdir /workspace/compiled_wheels
