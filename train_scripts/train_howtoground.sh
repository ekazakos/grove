#!/bin/bash
#SBATCH --job-name=GROVE
#SBATCH --time=24:00:00
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=standard-g
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

conda activate grove

NNODES=$SLURM_NNODES

LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --rdzv_id=$SLURM_JOB_ID \
    --max_restarts 3 \
    --tee 3 \
    "

export OMP_NUM_THREADS=1

export PYTHONPATH=$HOME/mmcv:$PYTHONPATH
export PYTHONPATH=$HOME/grove:$PYTHONPATH

$LAUNCHER train.py --lora_r 0 --lr 5e-5 --pretrained --version MBZUAI/GLaMM-GranD-Pretrained --ce_loss_weight 1 --giou_loss_weight 2 --temp_objectness_loss_weight 2 --train_mask_decoder --exp_name iGround --epochs 20 --steps_per_epoch 350 --train_ann_dir /home/train_annotations --train_keys /home/train_keys_deduplicated.pkl --val_ann_dir /home/val_annotations --val_keys /home/val_keys.pkl --video_dir /home/HowTo100M_small --grove_weights /home/grove_checkpoints/grove_pt_howtoground1m_ckpt.bin > iGround.log 2>&1