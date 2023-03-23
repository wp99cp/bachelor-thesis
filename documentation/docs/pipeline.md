# Working Pipeline

This page describes the pipelines used to train the model.

## High-Level Overview

The pipeline consists of the following steps:

1) Download the raw data from the server
2) Automatically annotate the raw data (using a mixture of existing models)
3) Train and validate the ML model

The pipeline can be executed using the following command:

```bash
bash helper_scripts/pipeline.sh pipeline-config.yml 
```

## Configuration

The pipeline is configured using the `config.yaml` file.

```yml
data_handling:
  # forces to download the raw data from the pf-server
  force_download: false
```

