################
# Mount data directories
################

export ANNOTATED_MASKS_DIR="$SCRATCH/data_sources/masks"
export EXTRACTED_RAW_DATA="$SCRATCH/data_sources/extracted_raw_data"

################
# Run the task
################

source "$TASK_DIR/pipeline.sh"

################
# Copy the produced Data
################

# Copy the results to the shared scratch space
echo "Copying the results to the shared scratch space"
echo "That is $SCRATCH/$SLURM_JOB_ID"
