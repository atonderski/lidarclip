#!/bin/bash
#
#SBATCH -N 1
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:4
#SBATCH --time 3-00:00:00
#SBATCH --output /mimer/NOBACKUP/groups/clippin/users/%u/logs/%j.out
#SBATCH -A SNIC2022-22-1004
#

export MASTER_PORT=$RANDOM

singularity exec --nv \
  --bind /mimer/NOBACKUP/groups/clippin/users/$USER:/workspace \
  --bind /mimer/NOBACKUP/groups/clippin/checkpoints:/checkpoints \
  --bind /mimer/NOBACKUP/groups/clippin/features:/features \
  --bind /mimer/NOBACKUP/groups/clippin/datasets/once:/my_data \
  --bind /mimer/NOBACKUP/groups/clippin/users/$USER/lidar-clippin/SST/mmdet3d/ops/sst/sst_ops.py:/sst/mmdet3d/ops/sst/sst_ops.py \
  --bind /mimer/NOBACKUP/groups/clippin/users/$USER/lidar-clippin/SST/mmdet3d/models/backbones/sst_v1.py:/sst/mmdet3d/models/backbones/sst_v1.py \
  --bind /mimer/NOBACKUP/groups/clippin/users/$USER/lidar-clippin/SST/mmdet3d/models/backbones/sst_v2.py:/sst/mmdet3d/models/backbones/sst_v2.py \
  --bind /mimer/NOBACKUP/groups/clippin/users/$USER/lidar-clippin/SST/mmdet3d/models/voxel_encoders/utils.py:/sst/mmdet3d/models/voxel_encoders/utils.py \
  --pwd /workspace/lidar-clippin/ \
  --env PYTHONPATH=/workspace/lidar-clippin/ \
  /mimer/NOBACKUP/groups/clippin/containers/lidar-clippin.sif \
  python3 -u train.py --data-dir=/my_data --checkpoint-save-dir=/checkpoints $@

#
#EOF
