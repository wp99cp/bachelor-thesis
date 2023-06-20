
################
# Run the task
################

current_dir=$(pwd)
cd ../../
source "$BASE_DIR/utilities/infer.sh"
cd $current_dir

################
# Copy the produced Data
################

# Copy the results to the shared scratch space
echo "Copying the results to the shared scratch space"
echo "That is $SCRATCH/$SLURM_JOB_ID"
