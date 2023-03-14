# This script is run before the actual task is executed on Euler. This script is running
# on a compute node, so it can use the local scratch space. It is used to prepare the
# task for execution, i.g. copy the training data to the local scratch space, compile
# the code, etc.

# Print CPU information
lscpu
echo -e "\n\n"

# If available, print GPU information
nvidia-smi
echo -e "\n\n"

# Copy the training data to the local scratch space
cp "$DATASET" "$TMPDIR"
echo "Copied the training data to $TMPDIR"

# Unzip the training data
mkdir -p "$TMPDIR/data"
unzip "$TMPDIR/dataset.zip" -d "$TMPDIR/data"
echo "Unzipped the training data"

# Update the environment variable DATA_DIR to point to the local scratch space
export DATA_DIR="$TMPDIR/data"
echo "The training data is now located in $DATA_DIR"
ls -l "$DATA_DIR"

# Export the task directory to the environment
export TASK_DIR="$PWD"
echo "The task directory is $TASK_DIR"

# Export the path to the LOG_DIR to the environment
export LOG_DIR="$TMPDIR/log"
mkdir -p "$LOG_DIR"
echo "The log directory is $LOG_DIR"

echo -e "\n\n============================\n\n"
