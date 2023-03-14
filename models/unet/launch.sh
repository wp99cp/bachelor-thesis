#!/bin/bash

#SBATCH -n 1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=512
#SBATCH --open-mode=truncate
#SBATCH --output=/cluster/scratch/pucyril/%j/log/slurm-output.out
#SBATCH --error=/cluster/scratch/pucyril/%j/log/slurm-error.out
#SBATCH --tmp=3000
#SBATCH --gpus=1
#SBATCH --mail-type=END

# load modules
module load python/3.7.4
module load cuda/11.7.0
module list | echo

# Dataset Config
export DATASET="/cluster/scratch/pucyril/dataset.zip"

# start the job
source job.sh
