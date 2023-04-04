# Working Pipeline

This page describes the pipelines used to train the model.

## High-Level Overview

The pipeline consists of the following steps:

1) Download the raw data from the server. On Euler this step is always skipped.
2) Automatically annotate the raw data (using a mixture of existing models)
3) Create the training dataset
4) Train and validate the ML model

The pipeline can be executed using the following command:

```bash
bash helper-scripts/pipeline/pipeline.sh pipeline-config.yml 
```

## Configuration

The pipeline is configured using the `pipeline-config.yml` file.

### What's the difference between `pipeline-config.yml` and the `config.py` file of each step?

The `pipeline-config.yml` file is used to configure the pipeline and it's steps. The `config.py` file of each step
is used to configure the step itself. The later goes much more into detail and is used for example to configure
the model architecture and its hyperparameters.

## Special Case Euler

On Euler the pipeline is slightly different. Step (1) is never executed. Instead, extracted data must be provided
as a directory. The directory must contain the following structure:

```
- extracted_data
 | - 32TNS-auxiliary_data
 | - ExoLabs_classification_S2
 | - S2**
 | - ...
```

Optionally (i.g. if step (2) is skipped). You must provide the annotated masks as a directory. The directory must
contain the following structure:

```
- annotated_masks
 | - 2021...
 | - ....
```

