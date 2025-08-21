#!/bin/bash
#SBATCH --job-name=embed_tokens
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --partition=standard-g
#SBATCH --mem 256G

GROVE_WEIGHTS=$1
TOKEN_EMBEDDINGS_PATH=$2

export PYTHONPATH=$HOME/mmcv:$PYTHONPATH
export PYTHONPATH=$HOME/grove:$PYTHONPATH

python embed_tokens.py --grove_weights $GROVE_WEIGHTS --token_embeddings_path $TOKEN_EMBEDDINGS_PATH > embed_tokens.log 2>&1
