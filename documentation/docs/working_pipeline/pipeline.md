# The End-to-End-Pipeline

This page describes the end-to-end pipelines used to download the data, train the model, and run inference on new
images. The pipeline can also handel the verification of the model and the creation of the training dataset.

- [High-Level Overview](#high-level-overview)
- [Configuration of the Pipeline](#configuration)
- [Data Sources and Mask Creation](/docs/datasources/datasources)

## High-Level Overview

The pipeline consists of the following steps:

1) Download the raw data from the server. On Euler this step is always skipped.
2) Automatically annotate the raw data (using a mixture of existing models).
3) Creation of the training / verification dataset.
4) Training of the model.
5) Inference on new images.
6) Testing of the model.

The pipeline can be executed using the following command:

```bash
bash helper-scripts/pipeline/pipeline.sh pipeline-config.yml 
```

## Configuration

The pipeline is configured using the `./pipeline-config.yml` file.

### Difference between `pipeline-config.yml` and `config.py`?

The `pipeline-config.yml` file is used to configure the pipeline and it's steps. The `config.py` file of each step
is used to configure the step itself. The later goes much more into detail and is used for example to configure
the model architecture and its hyperparameters.

## Special Case Euler

On Euler the pipeline is slightly different. Step (1) is never executed. Instead, extracted data must be provided
as a directory. See [how to copy data to Euler](/docs/nice_to_know/euler.html#copy-dataset-to-euler) for more
information.

The directory must contain the following structure:

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

### Start the Pipeline on Euler

The pipeline should never be started manually on Euler. Instead, use the provided ansible script. See
[Using Ansible to Automate Submission](/docs/nice_to_know/euler.html#using-ansible-to-automate-submission) for more
information.