# After the task is executed after the main task has finished, this script is run to
# clean up the task. This script is running on a compute node, so it can use the local
# scratch space. It is used to clean up the task, i.g. copy the results to the
# shared scratch space, send a notification etc.

echo -e "\n\n============================\n\n"

# Copy the results to the shared scratch space
echo "Copying the results to the shared scratch space"
echo "That is $SCRATCH/$SLURM_JOB_ID"

mkdir -p "$SCRATCH/$SLURM_JOB_ID"
cp -r "$RESULTS_DIR" "$SCRATCH/$SLURM_JOB_ID"
cp -r "$LOG_DIR" "$SCRATCH/$SLURM_JOB_ID"

echo -e "\n"

# Cleanup: delete the BASE_DIR
wait
echo "Deleting the BASE_DIR: $BASE_DIR"
rm -rf "$BASE_DIR"