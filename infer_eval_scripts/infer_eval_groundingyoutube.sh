#!/bin/bash

GROVE_WEIGHTS_DIR=$1
TOKEN_EMBEDDINGS_PATH=$2
OUTPUT_PATH=$3
GT_FILE_PATH=$4
VIDEO_DIR_PATH=$5

if [ -d "$GROVE_WEIGHTS_DIR" ]; then
    cd "$GROVE_WEIGHTS_DIR"
    python zero_to_fp32.py ./ pytorch_model.bin
    GROVE_WEIGHTS="$GROVE_WEIGHTS_DIR/pytorch_model.bin"
    cd "$HOME/grove"
elif [[ "$GROVE_WEIGHTS_DIR" == *.bin ]]; then
    GROVE_WEIGHTS="$GROVE_WEIGHTS_DIR"
else
    echo "Error: GROVE_WEIGHTS_DIR is neither a directory nor a .bin file."
    exit 1
fi

# Submit job for embedding tokens
job_id1=$(sbatch --parsable embed_tokens.sh $GROVE_WEIGHTS $TOKEN_EMBEDDINGS_PATH)


# Submit job for inference
job_id2=$(sbatch --parsable --dependency=afterok:$job_id1 infer_eval_scripts/infer_groundingyoutube.sh $GROVE_WEIGHTS $TOKEN_EMBEDDINGS_PATH $OUTPUT_PATH $GT_FILE_PATH $VIDEO_DIR_PATH)

# Submit job for evaluation
sbatch --dependency=afterok:$job_id2 infer_eval_scripts/eval_groundingyoutube.sh $OUTPUT_PATH $GT_FILE_PATH $TOKEN_EMBEDDINGS_PATH