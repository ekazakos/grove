#!/bin/bash
#SBATCH --job-name=eval_anet
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

for file in $(find $STANFORD_CORENLP_PATH -name "*.jar"); do
    export CLASSPATH="$CLASSPATH:$(realpath $file)"
done

python eval_anet.py \
    -s $PRED_FILE_PATH \
    -r "$GT_FILE_PATH/anet_entities_cleaned_class_thresh50_trainval.json" \
    --split_file "$GT_FILE_PATH/split_ids_anet_entities.json" \
    --split "validation" --eval_mode "gen" \
    --loc_mode "all" -v

python eval_anet.py \
    -s $PRED_FILE_PATH \
    -r "$GT_FILE_PATH/anet_entities_cleaned_class_thresh50_trainval.json" \
    --split_file "$GT_FILE_PATH/split_ids_anet_entities.json" \
    --split "validation" --eval_mode "gen" \
    --loc_mode "loc" -v
    

rm $TOKEN_EMBEDDINGS_PATH