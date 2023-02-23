#!/bin/bash

#SBATCH -n 1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=512
#SBATCH --open-mode=truncate
#SBATCH --tmp=3000
#SBATCH --gpus=1
#SBATCH --mail-type=END

module load python
module load cuda/11.7.0

source job.sh
