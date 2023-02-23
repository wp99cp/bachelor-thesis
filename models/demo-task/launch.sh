#!/bin/bash

#SBATCH -n 1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=512
#SBATCH --output=slurm_{job_id}_output.txt
#SBATCH --error=slurm_{job_id}_error.txt
#SBATCH --open-mode=truncate

module load python
module load cuda/11.7.0

source job.sh
