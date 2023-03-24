################
# Mount data directories
################

export ANNOTATED_MASKS_DIR="$SCRATCH/annotated_masks"
extort EXTRACTED_RAW_DATA="$SCRATCH/extracted_data"

################
# Run the task
################

source "$TASK_DIR/pipeline.sh"

################
# Copy the produced Data
################

# create a zip from the ANNOTATED_MASKS_DIR
echo "Create a zip from the $ANNOTATED_MASKS_DIR"
tar -czf "$ANNOTATED_MASKS_DIR/annotated_masks.tar.gz" "$ANNOTATED_MASKS_DIR"

# create a zip from the $DATASET_DIR
echo "Create a zip from the $DATASET_DIR"
tar -czf "$DATASET_DIR/dataset.tar.gz" "$DATASET_DIR"

# Copy the results to the shared scratch space
echo "Copying the results to the shared scratch space"
echo "That is $SCRATCH/$SLURM_JOB_ID"

mkdir -p "$SCRATCH/$SLURM_JOB_ID/dataset"
cp -r "$DATASET_DIR/dataset.tar.gz" "$SCRATCH/$SLURM_JOB_ID/dataset/"

mkdir -p "$SCRATCH/$SLURM_JOB_ID/annotated_masks"
cp -r "$ANNOTATED_MASKS_DIR/annotated_masks.tar.gz" "$SCRATCH/$SLURM_JOB_ID/annotated_masks/"