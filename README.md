# Inverse Folding
This repository contains all code and instructions for setting up an enviroment to preform inverse folding using 3DI strucutres and natural language output. This was adapted from the Pannot LLM (https://github.com/Antoninnnn/Pannot/blob/master/README.md)

## Command to setup enviroment
**For Grace (TAMU)**
```
module purge

ml CUDA/11.8.0 Anaconda3

```

**For Conda Enviroment**

```
git clone --branch master https://github.com/Antoninnnn/Pannot.git # this get you the default Pannot github 

cd Pannot

conda create -n pannot-dev python=3.10 -y #Make sure you are using a new enviroment

conda activate pannot-dev

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade pip

pip install -e . # will install everything that is in pyproject.toml. Make sure you are in the Pannot directory

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install torch-geometric

pip install -e .[train]

pip install "flash-attn<=2.5.6" --no-build-isolation

pip install transformers==4.44.0
```

**Files to be downloaded from huggingface**
```
#First be in the Pannot directory:
# Then create a directory inorder to store the esm-2 model in

```
