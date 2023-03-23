# Working Pipeline

This page describes the pipelines used to train the model.

## High-Level Overview

The pipeline consists of the following steps:

1) Download the raw data from the server
2) Automatically annotate the raw data (using a mixture of existing models)
3) Create the training dataset
4) Train and validate the ML model

The pipeline can be executed using the following command:

```bash
bash helper-scripts/pipeline.sh pipeline-config.yml 
```

## Configuration

The pipeline is configured using the `config.yaml` file.

```yml
data_handling:

  # forces to download the raw data from the pf-server (1 or 0)
  force_rsync: 0

  # use single quotes for the dates and the array syntax
  s2_dates: ['20211008T101829', ..., '20210406T102021']

annotation:

  # use the auto annotator to create the ground truth (1 or 0)
  auto_annotation: 0
```

