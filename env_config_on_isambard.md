# Environment Configuration on Isambard
## Basic Libraries
Use ```module avail``` to check available python, cuda, cudatoolkit libraries pre-installed on the cluster and load them with ```module load <library_name>```.

## Create A Virtual Env (venv) for Your Project
Run ```python -m venv <venv_name>``` (under your project folder).

After venv is created, run ```source <venv_name>/bin/activate``` to switch to it.

## Install torch

Run ```pip install torch --index-url https://download.pytorch.org/whl/cu126``` (yes I know we just loaded cuda12.2 but we need cu126 to get torch to work well with cuda on isambard).

## Install bitsandbytes (for LLMs quantization)
### Probable Pre-requisite - Upgrade cmake (without sudo)
Download cmake source code using ```wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-aarch64.tar.gz```

Unzip it to a folder and add the new cmake path into ```PATH=$HOME/<path_to_cmake>/bin:$PATH```

Run ```cmake --version``` to check cmake version.

### Compile bitsandbytes locally (have to)
Set environment variable to set a special cuda version for bitsandbytes.
```bash
BNB_CUDA_VERSION=122
```

Download source code and compile it locally.
```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install . # DON'T use -e here!
```
Verify Installation
```bash
python -m bitsandbytes
```
Set environment variable ```BNB_CUDA_VERSION=122``` whenever using bitsandbytes. For example you load a large language model with quantization in ```train.py```, run it with 
```bash
BNB_CUDAVERSION=122 python train.py
``` 
Or write ```export BNB_CUDA_VERSION=122``` in your ```~/.bashrc``` and run
```bash
source ~/.bashrc
```
so you don't need to add it before your command everytime.

Now your environment should be ready to train or test LLMs.