# ETH Provides me a Desktop Machine

ETH provides me a desktop machine with a GPU. I can access it via SSH and run my experiments there.

```bash
ssh pucyril@pf-pc20.ethz.ch
```

All my files are stored at `/scratch2/pucyril/`. I can mount this folder on my local machine via SSHFS.

```bash
sshfs pucyril@pf-pc20.ethz.ch:/scratch2/pucyril /mnt/pf-pc20
```

## Run Experiments

::: details
Some pip modules must be manually installed on the remote machine. You can do this by running the following command:

```bash
pip install imutils
pip install pytorch-model-summary
```

:::

1) Pull the latest version of the repository.

    ```bash
    cd /scratch2/pucyril/bachelor-thesis
    git pull
    ```

2) Export the necessary environment variables.

    ```bash
    export TMPDIR=/scratch2/pucyril/tmp
    export DATA_DIR=/scratch2/pucyril/data
    export RESULTS_DIR=/scratch2/pucyril/results
    ```

3) Run the experiment. This can be done by running the python script directly.

   ```bash
   scp dataset.zip pucyril@pf-pc20.ethz.ch:/scratch2/pucyril/data/dataset.zip
   ```

   Then on the remote machine:

   ```bash
    unzip /scratch2/pucyril/data/dataset.zip -d /scratch2/pucyril/data
    ```

