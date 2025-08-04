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
