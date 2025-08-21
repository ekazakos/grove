#!/bin/bash
#SBATCH --job-name=InferenceGROVE
#SBATCH --time=2-00:00:00
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=standard-g
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G

GROVE_WEIGHTS=$1
TOKEN_EMBEDDINGS_PATH=$2
OUTPUT_PATH=$3
GT_FILE_PATH=$4
VIDEO_DIR_PATH=$5
TEMP_OBJECTNESS_THRESHOLD=$6

conda activate grove

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

NNODES=$SLURM_NNODES

LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 3 \
    --rdzv_id=$SLURM_JOB_ID \
    --tee 3 \
    "

export OMP_NUM_THREADS=1

export PYTHONPATH=$HOME/mmcv:$PYTHONPATH
export PYTHONPATH=$HOME/grove:$PYTHONPATH

$LAUNCHER infer_iground.py --grove_weights $GROVE_WEIGHTS --token_embeddings $TOKEN_EMBEDDINGS_PATH --video_info $GT_FILE_PATH --output_path $OUTPUT_PATH --temp_objectness_threshold $TEMP_OBJECTNESS_THRESHOLD --video_dir $VIDEO_DIR_PATH > infer.log 2>&1