#!/bin/bash
#SBATCH --job-name=eval_groungingyoutube
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --partition=standard-g
#SBATCH --mem 256G

PRED_FILE_PATH=$1
GT_FILE_PATH=$2
TOKEN_EMBEDDINGS_PATH=$3

conda activate grove

python eval_youcookinteractions.py \
    --predictions $PRED_FILE_PATH \
    --ground_truth $GT_FILE_PATH \
    --dataset groundingyoutube

rm $TOKEN_EMBEDDINGS_PATH