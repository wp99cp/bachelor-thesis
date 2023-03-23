################
# Copy and extract data
################

# Dataset Config
export RAW_DATASET="/cluster/scratch/pucyril/raw_data.zip"

# Copy the training data to the local scratch space
echo "Copy the raw data to $TMPDIR..."
cp "$RAW_DATASET" "$TMPDIR"
echo "Copied the raw data to $TMPDIR"

# Unzip the training data
mkdir -p "$DATA_RAW_DIR"
echo "Unzip the raw data..."
unzip -q "$TMPDIR/raw_data.zip" -d "$DATA_RAW_DIR"
echo "Unzipped the raw data"

################
# Run the task
################

echo "All further logs are saved in $LOG_DIR/python.log"

source pipeline.sh

################
# Copy the produced Data
################

# create a zip from the $DATASET_DIR
echo "Create a zip from the $DATASET_DIR"
zip -r "$DATASET_DIR.zip" "$DATASET_DIR/dataset.zip"

# Copy the results to the shared scratch space
echo "Copying the results to the shared scratch space"
echo "That is $SCRATCH/$SLURM_JOB_ID"

mkdir -p "$SCRATCH/$SLURM_JOB_ID/dataset"
cp -r "$DATASET_DIR/dataset.zip" "$SCRATCH/$SLURM_JOB_ID/dataset/"