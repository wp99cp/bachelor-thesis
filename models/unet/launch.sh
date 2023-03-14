#!/bin/bash

#SBATCH -n 8
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=4096
#SBATCH --open-mode=truncate
#SBATCH --output=/cluster/scratch/pucyril/%j/log/slurm-output.out
#SBATCH --error=/cluster/scratch/pucyril/%j/log/slurm-error.out
#SBATCH --tmp=3000
#SBATCH --gpus=1
#SBATCH --mail-type=END

# load modules
module load gcc/8.2.0
module load python_gpu/3.10.4
module list 2>&1 | tee "$TMPDIR/log/module-list.log"

# Dataset Config
export DATASET="/cluster/scratch/pucyril/dataset.zip"

# start the job
source job.sh
