# Data Pre-Processing

The data pre-processing is mostly done using the [pipeline](/docs/working_pipeline/pipeline). However, some manual steps
are also needed. This page gives an overview of the data pre-processing steps.

- [What the Pipeline can do](#what-the-pipeline-can-do)
- [Hand Annotation](/docs/pre-processing/hand_annotations)
- [Automated Annotation](/docs/pre-processing/automated_annotations)
- [Manual Cleanup](/docs/pre-processing/clean_up_masks)
- [Data Augmentation and Sampling](/docs/pre-processing/augmentation_and_sampling)

## What the Pipeline can do:

The first three steps of the pipeline are dedicated to data handling, pre-processing and dataset creation.
Those steps are disabled for later runs, as the data is already pre-processed and the dataset is already created and
cached.

However, if you want to re-run those steps, you can do so by setting the corresponding flags in
the `pipeline-config.yml` file.

```yml

# this section describes the data handling, i.g. the preparation 
# of the data: download and extraction, most of those settings 
# will be ignored on euler
data_handling:

  # [ignored on euler] forces to download the raw data from 
  # the pf-server (1 or 0) this will delete all old data and 
  # freshly download the s2_dates listed below
  # currently all data is manually downloaded to the pf-pc20
  # this was done using "cp -r /home/pf/pfstud/nimbus/download_data/* \
  # /scratch2/pucyril/bachelor-thesis/data/raw"
  force_rsync: 1
  # [...]

# this section describes the preprocessing of the data
# e.g. if the masks should be created automatically or if
# some manual annotations should be used
annotation:

  # use the auto annotator to create the ground truth (1 or 0)
  # currently this computes the s2cloudless predictions 
  # which is very slow then combines them with the 
  # ExoLab predictions
  auto_annotation: 1
  # [...]

# This section describes the training and 
# validation dataset creation/loading
dataset:

  # forces a recreation of the dataset (1 or 0)
  # this deletes the dataset folder and recreates
  # it using the current raw data will be ignored 
  # if create_on_the_fly is enabled
  recreate_dataset: 1
  # [...]

```
