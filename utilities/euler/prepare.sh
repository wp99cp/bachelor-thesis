# This script is run before the actual task is executed on Euler. This script is running
# on a compute node, so it can use the local scratch space. It is used to prepare the
# task for execution, i.g. copy the training data to the local scratch space, compile
# the code, etc.

# Print CPU information
lscpu
echo -e "\n\n"

# If available, print GPU information
nvidia-smi
echo -e "\n"
nvidia-smi --query-gpu=name --format=csv,noheader
lscpu | sed -nr '/Model name/ s/.*:\s*(.*) @ .*/\1/p'
echo -e "\n\n"

# print git info
echo "Git commit hash: $(git log --pretty=format:'%h' -n 1)"
echo "View the commit online: https://github.com/wp99cp/bachelor-thesis/commit/$(git log --pretty=format:'%h' -n 1)"

# Set the environment variable RUNS_ON_EULER to 1
# this allow us to do euler specific things in the pipeline
# or ignore, e.g. delete commands, on euler
export RUNS_ON_EULER=1

# create tmp dir inside the tmp dir
export TMP_DIR="$TMPDIR/tmp"
mkdir -p "$TMP_DIR"

# Update the environment variable DATA_DIR to point to the local scratch space
export DATA_SENTINEL2=$SCRATCH/data_sources/data/sentinel2
export DATA_LANDSAT8=$SCRATCH/data_sources/data/landsat8
export MODEL_SAVE_DIR=$SCRATCH/data_sources/model_checkpoints
# create a RESULTS_DIR
export RESULTS_DIR="$TMPDIR/results"
mkdir -p "$RESULTS_DIR"
echo "The results directory is $RESULTS_DIR"

# Export the task directory to the environment
export TASK_DIR="$PWD"

# get dirname without ...
export BASE_DIR="$(realpath "$TASK_DIR/../..")"
export PYTHONPATH="$BASE_DIR"
echo "PYTHONPATH: $PYTHONPATH"

echo "BASE_DIR: $BASE_DIR"
echo "The task directory is $TASK_DIR"

# Export the path to the LOG_DIR to the environment
export LOG_DIR="$SCRATCH/$SLURM_JOB_ID/log"
mkdir -p "$LOG_DIR"
echo "The log directory is $LOG_DIR"

echo -e "\n\n============================\n\n"
