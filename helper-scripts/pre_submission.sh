# This script is run before a the task is submitted on Euler. It is used to
# prepare the task for submission, compile the code, transfer the training
# data if changed, load the modules, etc.

# Load the modules
echo "Loading modules required for build"
module list &> /dev/null || source /cluster/apps/modules/init/bash

module load python/3.10.4
