# Inverse Folding
This repository contains all code and instructions for setting up an enviroment to preform inverse folding using 3DI strucutres and natural language output. This was adapted from the Pannot LLM (https://github.com/Antoninnnn/Pannot/blob/master/README.md)

The code will read a csv file that contains the name, 3di sequence and amino acid sequence of the protien. Then it will output a csv that will contain a number of predicted sequences, their lengths and expected lengths. Make sure you have an output path for the csv. 

## Dataset:
"CASP15 is the dataset I used to experiment with the model, but you're welcome to try others as well. Keep in mind that the prompt has a token limit of 2048, which restricts how many training examples you can include. If this limit is exceeded, the model will forget the earliest parts of the input. So far, CASP15 has worked well with 7 training examples and 1 query. You can experiment further to determine how many examples your setup can handle. CATH4.2 and CATH4.3 are also compatible datasets worth exploring. This is the same dataset used in the ProteinInvBench benchmark: https://github.com/A4Bio/ProteinInvBench


## For creating an interactive inference (running the Inverse_Folding_LLM.py):
```python

srun --partition=gpu --gres=gpu:a100:1 --nodes=1 --ntasks=2 --cpus-per-task=4 --mem=96G --time=08:00:00 --pty bash
#wait for the srun to finish allocation then continue

module purge

module load CUDA/11.8.0 Anaconda3
# eval "$(conda shell.bash hook)"


# Set CUDA environment variables on Grace HPRC of TAMU
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH

# Set Hugging Face cache directory to a writable location
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1


#Set the Torch cache directory in the $SCRATCH

export TORCH_HOME=$SCRATCH/.cache/torch

# # Create the directory if it doesn't exist
# mkdir -p $TRANSFORMERS_CACHE


source activate pannot-dev1

# # The reason I deactivate and activate again is that 
# # I want to make sure the python is used in the environment,
# # not the default python in the system.(in sw/...)
# # (the problem would occur when i activate and directly call python)
conda deactivate 

source activate pannot-dev1


# Example: Pannot pretraining script (multimodal: protein sequence + structure)
# Be sure to set these environment variables or modify inline:

MODEL_VERSION=Meta-Llama-3.1-8B-Instruct
PROMPT_VERSION=plain

# Customize these:
DATA_PATH=/scratch/user/jawadnelhassan2005/LLM/Pannot/data/opi/OPI_full_1.61M_train_converted.jsonl #<-- will be different for you. Just give it a directory to the OPI_full_1.61M_train_converted.jsonl
# DATA_PATH=$SCRATCH/TAMU/PhD/Pannot/data/opi/OPI_full_1.61M_train_first_10000.json
PRET_MODEL_DIR=./checkpoints/pannot-${MODEL_VERSION}-pretrain-v00
SEQ_TOWER=ESM
STR_TOWER=ESMIF

python Inverse_folding_llm.py
```

## Commands to setup enviroment
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
## To run scatterplot.py, you must execute it locally on your computer rather than on the HPRC. Here's what you'll need to get started:

```python
pip install pandas

pip install matplotlib
```


## changes to some python files
**You must have to update the directories in the code to match your directories inorder for it to work properly.**
You will have to go into the following directory (/scratch/user/.../Pannot/pannot/model/multimodel_encoder). Inside this directory you must update the esm_seqeunce_encoder.py code to the code bellow:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PretrainedConfig
 
class ESMSeqTower(nn.Module):
    def __init__(
        self,
        model_name: str = None,
        args=None,
        delay_load: bool = False,
        no_pooling: bool = True,
    ):
        super().__init__()
        self.is_loaded = False
        self.args = args
 
        # Default to local model path for offline use
        self.local_model_dir = "/scratch/user/jawadnelhassan2005/LLM/Pannot/local_pretrained_encoders/esm2_t33_650M_UR50D"
        self.model_path = model_name or self.local_model_dir
 
        self.select_layer = getattr(args, 'mm_seq_select_layer', -1)
        self.pooling = getattr(args, 'mm_seq_select_feature', 'cls')  # 'cls' or 'mean'
        self.no_pooling = getattr(args, 'mm_seq_no_pooling', no_pooling)
 
        if not delay_load or getattr(args, 'unfreeze_mm_seq_tower', False):
            self.load_model()
 
    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.model_path} is already loaded. Skipping load.')
            return
 
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.encoder = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            output_hidden_states=True,
            device_map=device_map
        )
        self.encoder.requires_grad_(False)
        self.is_loaded = True
 
    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        if not self.is_loaded:
            self.load_model()
 
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
 
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
 
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(self.device)
 
        vocab_size = self.encoder.config.vocab_size
        assert (input_ids < vocab_size).all(), f"Token id out of range! Max: {input_ids.max().item()}, vocab_size: {vocab_size}"
 
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[self.select_layer]
 
        if self.no_pooling:
            return hidden_states
 
        if self.pooling == 'cls':
            return hidden_states[:, 0, :]
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_emb = torch.sum(hidden_states * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return sum_emb / counts
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")
 
    def tokenize(self, sequences, return_tensors='pt', padding=True, truncation=True, max_length=1024):
        if not self.is_loaded:
            self.load_model()
        return self.tokenizer(
            sequences,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )
 
    @property
    def dummy_feature(self):
        if self.no_pooling:
            return torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
 
    @property
    def dtype(self):
        return self.encoder.dtype if self.is_loaded else torch.get_default_dtype()
 
    @property
    def device(self):
        return next(self.encoder.parameters()).device if self.is_loaded else torch.device('cpu')
 
    @property
    def config(self):
        return self.encoder.config if self.is_loaded else PretrainedConfig.from_pretrained(
            self.model_path,
            local_files_only=True
        )
 
    @property
    def hidden_size(self):
        return self.config.hidden_size

```

Now go this directory (scratch/user/.../Pannot/pannot/) and update the subroutine called _format_message to this:
```python
def _format_message(self, message: Union[str, ProteinInput]) -> str:
    if isinstance(message, ProteinInput):
        fields = []
        if message.sequence:
            fields.append(f"<seq> {message.sequence} </seq>")
        if message.structure:
            fields.append(f"<str> {message.structure} </str>")
        if message.annotations:
            fields.append(f"<anno> {message.annotations} </anno>")
        return "\n".join(fields)
    return message
```


