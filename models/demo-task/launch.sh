#!/bin/bash

#SBATCH -n 64
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=512
#SBATCH --output=slurm_output.txt
#SBATCH --error=slurm_error.txt
#SBATCH --open-mode=truncate
#SBATCH --constraint=EPYC_7742

module load python
module load cuda/11.7.0

source job.sh
