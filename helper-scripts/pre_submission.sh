# This script is run before a the task is submitted on Euler. It is used to
# prepare the task for submission, compile the code, transfer the training
# data if changed, load the modules, etc.

# Load the modules
echo "Loading modules required for build"
# shellcheck disable=SC2046
module() { eval `/cluster/apps/modules/bin/modulecmd bash $*`; }
export -f module

MODULESHOME=/cluster/apps/modules
export MODULESHOME

module load python/3.10.4
