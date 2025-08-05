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
#First create an hf_cache directory in your scratch/user/your_netid directory:
#Second create a directory in your Pannot directory called local_pretrained_encoders
#Thirdly create another directory inside your Pannot directory called checkpoints
#NOW, prior to continuing you must have a huggingface account.
pip install -U "huggingface_hub[cli]"

huggingface-cli login

export HF_HOME=/scratch/user/'''your neid'''/hf_cache

huggingface-cli download facebook/esm2_t33_650M_UR50D

hf download Yining04/Pannot_v0.0 

python # we want to use the python console for this

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/esm2_t33_650M_UR50D",
    repo_type="model",
    local_dir="/scratch/jawadnelhassan2005/esm2_t33_650M_UR50D",
    local_dir_use_symlinks=False
)

repo_id = "Yining04/Pannot_v0.0"

destination = "/scratch/user/jawadnelhassan2005/LLM/building_enviroment/Pannot/checkpoints"

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=destination,
    local_dir_use_symlinks=False,
    allow_patterns=[
        "pannot-Meta-Llama-3.1-8B-Instruct-finetune-lora-v02/**",
        "pannot-Meta-Llama-3.1-8B-Instruct-pretrain-v02/**"
    ]
)
```
