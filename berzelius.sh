#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 3-00:00:00
#SBATCH --output /proj/nlp4adas/users/%u/logs/%j.out
#

singularity exec --nv --bind /proj/nlp4adas/users/$USER:/workspace \
  --bind /proj/nlp4adas/datasets/once:/my_data \
  --pwd /workspace/2022-d3tr/ \
  --env PYTHONPATH=/workspace/2022-d3tr/ \
  /proj/berzelius-2021-92/pytorch21_09.sif \
  python3 -u $@

#
#EOF
