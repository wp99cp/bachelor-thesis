#!/bin/bash

#SBATCH -n 1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=512
#SBATCH --open-mode=truncate
#SBATCH --output=/cluster/scratch/pucyril/%j/log/output.out
#SBATCH --error=/cluster/scratch/pucyril/%j/log/error.out
#SBATCH --tmp=3000
#SBATCH --gpus=1
#SBATCH --mail-type=END

# load modules
module load python
module load cuda/11.7.0
module list

# install dependencies
pip install --user imutils==0.5.4
pip install --user pytorch-model-summary==0.1.2

# Dataset Config
export DATASET="/cluster/scratch/pucyril/dataset.zip"

# start the job
source job.sh
