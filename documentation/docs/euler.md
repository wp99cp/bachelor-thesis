# Working with Euler

Euler is the HPC-Cluster of ETH.

## Creating a Task

A task is a folder containing a `task.sh` file. This file specifies the commands to be executed on the cluster. The task
folder can reference additional scripts, data, etc. that are required for the task.

### Environment Variables

- The data folder is mounted to `$DATA_DIR` on the cluster.
- The task folder is mounted to `$TASK_DIR` on the cluster.
- Logs and results should be saved in `$LOG_DIR`.

## Using Ansible to Automate Submission

Euler is using the [SLURM](https://slurm.schedmd.com/documentation.html) workload manager. You can submit a task via
the `sbatch` command. For this thesis we use ansible to automate the deployment submission of the tasks

::: tip
Ensure that you have set up passwordless SSH access to the cluster.
See [this guide](https://scicomp.ethz.ch/wiki/Accessing_the_clusters) for more information. In addition, you must have
configured an SSH key for accessing the GitHub repository.

That is:

```bash
eval $(ssh-agent -s)
ssh-add ~/.ssh/id_ed25519_euler
ssh-add ~/.ssh/id_ed25519_github
```

:::

::: tip
Run the following commands inside the `/helper-scripts/ansible` directory.
:::

To submit task, e.g. `demo-task`, to the cluster, you must follow the following steps:

**(1) Preparation**

::: details
Some pip modules must be manually installed on the cluster. You can do this by running the following command:

```bash
pip install imutils
pip install pytorch-model-summary
```

**Note:** You must run this command while loaded the correct python environment (e.g. `module load python_gpu/3.10.4`).

:::

1) Create a directory for your task containing a `task.sh` file and a `launch.sh` file.
   The launch file must source `job.sh`. See the following example:

```bash
#!/bin/bash

#SBATCH -n 8
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=4096
#SBATCH --open-mode=truncate
#SBATCH --output=/cluster/scratch/pucyril/%j/log/slurm-output.out
#SBATCH --error=/cluster/scratch/pucyril/%j/log/slurm-error.out
#SBATCH --tmp=3000
#SBATCH --gpus=1
#SBATCH --mail-type=END

# load modules
module load gcc/8.2.0
module load python_gpu/3.10.4

# report the modules
mkdir -p "$TMPDIR/log"
module list 2>&1 | tee "$TMPDIR/log/module-list.log"

# Dataset Config
export DATASET="/cluster/scratch/pucyril/dataset.zip"

# start the job
source job.sh
``` 

2) Commit and push your changes to the repository

**(2) Submission**

1) Run the following command to submit your task to the cluster:

```bash
ansible-playbook -i hosts.yml submit_euler.yml --extra-vars "task=models/demo-task"
```

::: details
If you have problem with ansible, ass `ANSIBLE_DEBUG=1` in front of the command.
:::

This checks out the latest version of your task from the repository and submits it to the cluster.
If you want to submit a specific version of your task, you can specify the `commit` parameter:

```bash
ansible-playbook -i hosts.yml submit_euler.yml --extra-vars "task=demo-task commit=1234567890"
```

2) You get notified via email when the task is finished. Results are saved under `$SCRATCH/<job.id>`. This means
   that the results are only available for a limited time.

## Copy Dataset to Euler

To copy the dataset to the cluster, you first need to create a zip folder containing the dataset.

```bash
zip -r dataset.zip .
```

Then you can use the `scp` command to copy the zip file to the cluster. The following command copies the zip file to the
cluster:

```bash
scp -r dataset.zip pucyril@euler.ethz.ch:/cluster/scratch/pucyril
```

Monitor your running tasks with the following command (inside `/cluster/scratch/pucyril/<jobId>/log`)

```bash
watch "squeue && echo '' && tail -25 slurm-output.out"
```

Once the task is finished, you can view some stats using the following command:

```bash
myjobs -j  <jobId>
```

## Access

To access the cluster, you need to be a member of the ETH domain. You can then use your ETH credentials to log in to the
cluster. The login node is `euler.ethz.ch`, you can log in via SSH:

```bash
ssh <username>@euler.ethz.ch
```

## Mount Scratch folder

To mount the scratch folder, you can use the following command:

```bash
sudo sshfs -o allow_other,default_permissions pucyril@euler.ethz.ch:/cluster/scratch/pucyril /mnt/euler
```