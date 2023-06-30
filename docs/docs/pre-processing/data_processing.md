# Data Pre-Processing

The data pre-processing is mostly done using the [training](/docs/working_pipeline/training). However, some manual steps
are also needed. This page gives an overview of the data pre-processing steps.

- [What the Pipeline can do](#what-the-pipeline-can-do)
- [Hand Annotation](/docs/pre-processing/hand_annotations)
- [Automated Annotation](/docs/pre-processing/automated_annotations)
- [Manual Cleanup](/docs/pre-processing/clean_up_masks)
- [Data Augmentation and Sampling](/docs/pre-processing/augmentation_and_sampling)

## What the Pipeline can do:

The first two steps of the training pipeline are dedicated to data handling, pre-processing and dataset creation.
Those steps are disabled for later runs, as the data is already pre-processed and the dataset is already created and
cached.

However, if you want to re-run those steps, you can do so by setting the corresponding flags in
the `train_config.yml` file.

```yml
# This section describes which masks should be used to train the model
# masks can be generated automatically based on existing algorithms
# or masks can be loaded from a directory
ground_truth_masks:

  # use the auto annotator to create the ground truth (1 or 0)
  # currently this computes the s2cloudless predictions which is very slow
  # then combines them with the ExoLab predictions and the water mask
  # automatic maks can only be created for S2 data not for L8
  auto_annotation: 0

# This section describes the training and validation dataset creation
dataset:

  # forces a recreation of the dataset (1 or 0)
  # this deletes the dataset folder and recreates it using the current 
  # raw data will be ignored if create_on_the_fly is enabled
  recreate_dataset: 0

  # limits the number of dates to be used for the dataset
  # if you wish to use all dates, just set this to 0
  limit_dates: 1
```
