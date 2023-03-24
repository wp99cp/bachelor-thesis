################
# Copy and extract data
################

# Dataset Config
RAW_DATASET="/cluster/scratch/pucyril/raw_data.zip"
ANNOTATED_MASKS="/cluster/scratch/pucyril/annotated_masks.zip"

# Copy the training data to the local scratch space
echo "Copy the raw data to $TMP_DIR..."
cp "$RAW_DATASET" "$TMP_DIR"
echo "Copied the raw data to $TMP_DIR"

echo "Copy the annotated masks to $TMP_DIR..."
cp "$ANNOTATED_MASKS" "$TMP_DIR"
echo "Copied the annotated masks to $TMP_DIR"

# Unzip the training data
mkdir -p "$DATA_DIR"
echo "Unzip the raw data..."
unzip -q "$TMP_DIR/raw_data.zip" -d "$DATA_DIR/"
echo "Unzipped the raw data"

mkdir -p "$ANNOTATED_MASKS_DIR"
echo "Unzip the annotated masks..."
unzip -q "$TMP_DIR/annotated_masks.zip" -d "$DATA_DIR/"
echo "Unzipped the annotated masks"

echo "The raw data is now located in $DATA_RAW_DIR:"
ls "$DATA_RAW_DIR"

# Delete the raw data zip
rm "$TMP_DIR/raw_data.zip"
rm "$TMP_DIR/annotated_masks.zip"

################
# Run the task
################

source "$TASK_DIR/pipeline.sh"

################
# Copy the produced Data
################

# create a zip from the ANNOTATED_MASKS_DIR
echo "Create a zip from the $ANNOTATED_MASKS_DIR"
tar -czf "$ANNOTATED_MASKS_DIR/annotated_masks.tar.gz" -C "$ANNOTATED_MASKS_DIR"

# create a zip from the $DATASET_DIR
echo "Create a zip from the $DATASET_DIR"
tar -czf "$DATASET_DIR/dataset.tar.gz" -C "$DATASET_DIR"

# Copy the results to the shared scratch space
echo "Copying the results to the shared scratch space"
echo "That is $SCRATCH/$SLURM_JOB_ID"

mkdir -p "$SCRATCH/$SLURM_JOB_ID/dataset"
cp -r "$DATASET_DIR/dataset.tar.gz" "$SCRATCH/$SLURM_JOB_ID/dataset/"

mkdir -p "$SCRATCH/$SLURM_JOB_ID/annotated_masks"
cp -r "$ANNOTATED_MASKS_DIR/annotated_masks.tar.gz" "$SCRATCH/$SLURM_JOB_ID/annotated_masks/"