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
bash helper-scripts/pipeline.sh pipeline-config.yml 
```

## Configuration

The pipeline is configured using the `config.yaml` file.

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

