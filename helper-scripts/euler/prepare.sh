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

# Set the environment variable RUNS_ON_EULER to 1
# this allow us to do euler specific things in the pipeline
# or ignore, e.g. delete commands, on euler
export RUNS_ON_EULER=1

# Update the environment variable DATA_DIR to point to the local scratch space
export DATA_DIR="$TMPDIR/data"
mkdir -p "$DATA_DIR"
echo "The training data is now located in $DATA_DIR"

export DATA_RAW_DIR="$DATA_DIR/raw"
mkdir -p "$DATA_RAW_DIR"
echo "The raw data is now located in $DATA_RAW_DIR"

export ANNOTATED_MASKS_DIR="$DATA_DIR/annotated_masks"
mkdir -p "$ANNOTATED_MASKS_DIR"
echo "The annotated masks are now located in $ANNOTATED_MASKS_DIR"

export MASKS_DIR="$DATA_DIR/masks"
mkdir -p "$MASKS_DIR"
echo "The masks are now located in $MASKS_DIR"

export DATASET_DIR="$DATA_DIR/dataset"
mkdir -p "$DATASET_DIR"
echo "The dataset is now located in $DATASET_DIR"

# create a RESULTS_DIR
export RESULTS_DIR="$TMPDIR/results"
mkdir -p "$RESULTS_DIR"
echo "The results directory is $RESULTS_DIR"

# Export the task directory to the environment
export TASK_DIR="$PWD"

# Extract the BASE_DIR using a regular expression that matches
# the pattern "/cluster/home/pucyril/bachelor-thesis-*"
if [[ $TASK_DIR =~ /cluster/home/pucyril/bachelor-thesis-.* ]]; then
    # Remove the uuid suffix from the BASE_DIR using bash string manipulation
    BASE_DIR="${BASH_REMATCH[0]%-*}"
    export BASE_DIR="$BASE_DIR"
else
    echo "Error: TASK_DIR is not a subdirectory of BASE_DIR"
    exit 1
fi

echo "BASE_DIR: $BASE_DIR"
echo "The task directory is $TASK_DIR"

# Export the path to the LOG_DIR to the environment
export LOG_DIR="$SCRATCH/$SLURM_JOB_ID/log"
mkdir -p "$LOG_DIR"
echo "The log directory is $LOG_DIR"

echo -e "\n\n============================\n\n"
