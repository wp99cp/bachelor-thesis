################
# Run the task
################

source "$TASK_DIR/infer.sh"

################
# Copy the produced Data
################

# Copy the results to the shared scratch space
echo "Copying the results to the shared scratch space"
echo "That is $SCRATCH/$SLURM_JOB_ID"
