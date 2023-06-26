#!/bin/bash

#SBATCH --time=45:59:00
#SBATCH -n 2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=0
#SBATCH --gres=gpumem:20g
#SBATCH --tmp=12G
#SBATCH --open-mode=truncate
#SBATCH --output=/cluster/scratch/pucyril/%j/log/slurm-output.out
#SBATCH --error=/cluster/scratch/pucyril/%j/log/slurm-error.out
#SBATCH --account=es_schin
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=END

# load modules
module load gcc/8.2.0
module load python_gpu/3.11.2
module load cuda/11.8.0

# report the modules
mkdir -p "$TMPDIR/log"
module list 2>&1 | tee "$TMPDIR/log/module-list.log"

# set the environment variables describing the system resources
export NUM_PROCESSES=8
export TOTAL_MEMORY=80 # we reserve some memory for the system

################
# Prepare the task
################

# start the job
source job.sh
