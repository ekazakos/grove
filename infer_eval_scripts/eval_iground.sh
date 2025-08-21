#!/bin/bash
#SBATCH --job-name=eval_iground
#SBATCH --account=project_465001678
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --partition=standard-g
#SBATCH --mem 256G

PRED_FILE_PATH=$1
GT_FILE_PATH=$2
TOKEN_EMBEDDINGS_PATH=$3
STANFORD_CORENLP_PATH=$4

conda activate grove

PRED_FILE_NAME=$(basename $PRED_FILE_PATH)
PRED_FILE_NAME="${PRED_FILE_NAME%.*}"

SAVE_DIR="./$PRED_FILE_NAME"

export CLASSPATH=$CLASSPATH:$STANFORD_CORENLP_PATH/stanford-corenlp-3.4.1-models.jar

python eval_iground.py --split test --pred_file_path $PRED_FILE_PATH --gt_file_path $GT_FILE_PATH --save_dir $SAVE_DIR --evaluation_mode per_video > $PRED_FILE_NAME.log 2>&1

rm $TOKEN_EMBEDDINGS_PATH
rm -r $SAVE_DIR