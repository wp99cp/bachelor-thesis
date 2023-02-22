#!/bin/bash

# After the task is executed after the main task has finished, this script is run to
# clean up the task. This script is running on a compute node, so it can use the local
# scratch space. It is used to clean up the task, i.g. copy the results to the
# shared scratch space, send a notification etc.

echo "this file is executed on the compute node, after the job has successfully finished"