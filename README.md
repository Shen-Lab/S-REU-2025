# Multilingual
This repository contains the code and setup instructions for building an environment to explore cross-domain learning on proteins and biomedical texts, as part of the Multilingual-Molecules: Cross-Domain Learning Across Proteins and Biomedical Texts project. This was adapted from the Pannot LLM (https://github.com/Antoninnnn/Pannot/blob/master/README.md)

This script loads a Pannot-LLaMA model, reads ProteinGym metadata, and for each selected protein CSV builds a 2-shot prompt to classify whether a mutation preserves or improves function (Yes/No). It prints accuracy metrics, and—if you set an output path—saves a CSV with per-row predictions (DMS_id, function, mutation, true label, predicted label, decoded text, mutated sequence).

## Dataset
"ProteinGym" was used in this project. Once you are in the ProteinGYm website, you should download DMS Assays-> Subsitutions, and DMS-> Subsitutions.

## Creating Interactive Interface for Multilingual
```
srun --partition=gpu --gres=gpu:a100:1 --nodes=1 --ntasks=2 --cpus-per-task=4 --mem=96G --time=08:00:00 --pty bash
cd $SCRATCH/Pannot

module purge
module load CUDA/11.8.0 Anaconda3

export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH
export HF_HOME=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/.cache/torch
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1


source /scratch/user/gaygysyz2003/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/user/gaygysyz2003/envs/lama
conda deactivate
conda activate /scratch/user/gaygysyz2003/envs/lama
MODEL_VERSION=Meta-Llama-3.1-8B-Instruct
PROMPT_VERSION=plain
export DATA_PATH=/scratch/user/gaygysyz2003/TAMU/PhD/Pannot/data/opi/OPI_full_1.61M_train.json
export PRET_MODEL_DIR=./checkpoints/pannot-${MODEL_VERSION}-pretrain-v00
export SEQ_TOWER=ESM
export STR_TOWER=ESMIF



/scratch/user/gaygysyz2003/envs/lama/bin/fewshot_with_wildtype.py 
```
## Set up environment
### For Grace HPRC(TAMU)
```
module purge

ml CUDA/11.8.0 Anaconda3

<!-- module load GCC/12.3.0 CUDA/11.8.0 Anaconda3 --> ## if you want to use the predefined module of grace, you can use this line(caution: there would probably be version problem!)
```
### Set the conda virtual environment
```
conda create -n pannot-dev python=3.10 -y
source activate pannot-dev
pip install --upgrade pip  # enable PEP 660 support
pip install -e . # install the package defined in pyproject.toml

# You would need torch-geometric for protein structure processing(Used in GVP Module)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install torch-geometric
```
### Install train
```
pip install -e .[train]
pip install "flash-attn<=2.5.6" --no-build-isolation
```
### Files to be downloaded from huggingface


```
#---------Some instructions-------------------
#First create an hf_cache directory in your scratch/user/your_netid directory:
#Second create a directory in your Pannot directory called local_pretrained_encoders
#Thirdly create another directory inside your Pannot directory called checkpoints
#NOW, prior to continuing you must have a huggingface account.
#---------------------------------------------
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
    local_dir="/scratch/gaygysyz2003/esm2_t33_650M_UR50D",
    local_dir_use_symlinks=False
)

repo_id = "Yining04/Pannot_v0.0"

destination = "/scratch/user/gaygysyz2003/LLM/building_enviroment/Pannot/checkpoints"

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


