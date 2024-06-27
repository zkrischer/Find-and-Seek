# Installation Instruction

We recommend setting up the environment using Miniconda or Anaconda. We have tested the code on Linux with Python 3.10, torch 1.12.1, and cuda 11.6, but it should also work in other environments. If you have trouble, feel free to open an issue.

### Step 1: create an environment
Clone this repo:
```shell
git clone https://github.com/ywyue/AGILE3D.git
cd AGILE3D
```

```shell
conda create -n agile3d python=3.10
conda activate agile3d
```
### Step 2: install pytorch
```shell
# adjust your cuda version accordingly!
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```
### Step 3: install Minkowski
3.1 Prepare:
```shell
conda install openblas-devel -c anaconda
```
3.2 Install:
```shell
# adjust your cuda path accordingly!
export CUDA_HOME=/usr/local/cuda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```
We found the above steps are the easiest way to install Minkowski. If you run into issues, please also refer to their [official instructions](https://github.com/NVIDIA/MinkowskiEngine#installation).
### Step 4: install other packages
```shell
pip install open3d
```
