# Training

::: tip Start Training

You can start training with the following command:

```bash
bash utilities/train.sh [--config-file ./train_config.yml]
```

:::

## Configuration using YML and config.py

All three pipelines (training, inference, and testing) can be configured using two files. On the one hand, there
is the corresponding `*_config.yml` file, which is used to configure the pipeline itself, i.g. configure what
the pipeline should do. On the other hand, there is the `config.py` file, which is used to configure the python code,
set constants, etc.

The training pipeline can be configured using within the `train_config.yml` file.
