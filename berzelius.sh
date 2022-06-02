#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 3-00:00:00
#SBATCH --output /proj/nlp4adas/users/%u/logs/%j.out
#SBATCH -A berzelius-2022-117
#

singularity exec --nv --bind /proj/nlp4adas/users/$USER:/workspace \
  --bind /proj/nlp4adas/datasets/once:/my_data \
  --pwd /workspace/lidar-clippin/ \
  --env PYTHONPATH=/workspace/lidar-clippin/ \
  /proj/nlp4adas/containers/lidar-clippin.sif \
  bash -c "pip install pytorch_lightning && pip install --upgrade numpy && python3 -u train.py --data-dir=/my_data $@"

#
#EOF
