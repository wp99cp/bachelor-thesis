# Unet

Resources for Python Code

- [milesial/Pytorch-UNet/](https://github.com/milesial/Pytorch-UNet/tree/master)
- [U-Net: Training Image Segmentation Models in PyTorch](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)

## Train the Model

Using the [training pipeline](/docs/working_pipeline/training) the model can be trained.

## Model Architecture and Details

The model architecture is based on
the [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). The model is
implemented in `/src/models/unet/model/Model.py`. The code is based on https://github.com/milesial/Pytorch-UNet but
adjusted for supporting the high dimensional input data. The config file in `/src/models/unet/config/config.py` contains
the model configuration.

![Model Architecture](./../images/unet_architecture.png)

## Results

The following models were trained using the adjusted pre-processing and image splitting pipeline.
The new pipeline has updated the normalization and clipping of the data and is therefore not compatible with older
models.

::: tip Best Performing Algorithm

The best performing algorithm and the one used for evaluation is `a82e3cd` with the checkpoint of epoch
30: `unet_a82e3cd_ep30.pth`.

:::

::: info Git Commit ([a82e3cd](https://github.com/wp99cp/bachelor-thesis/commit/a82e3cd)) - 03.05.2023

Training ad described in the final report: using all 36 scenes, 30/70 percentile normalization. And trained for 60
epochs.

### Training results

![a82e3cd_training.png](./../images/a82e3cd_training.png)

```text
Epoch: 30, train_loss: 0.049460, test_loss: 0.040753, EStop: 0/30
Metrics for validation of epoch:  30
 - pixel_accuracy___background: 0.9772
 - union_over_inter_background: 0.9431
 - dice_coefficient_background: 0.9711
 - pixel_accuracy___snow: 0.9828
 - union_over_inter_snow: 0.9424
 - dice_coefficient_snow: 0.9708
 - pixel_accuracy___clouds: 0.9880
 - union_over_inter_clouds: 0.9513
 - dice_coefficient_clouds: 0.9755
 - pixel_accuracy___water: 1.0002
 - union_over_inter_water: 0.7838
 - dice_coefficient_water: 0.8665
```

and for the final epoch

```text
Start training epoch 60...
Epoch: 60, train_loss: 0.049416, test_loss: 0.041064, EStop: 24/30
Metrics for validation of epoch:  60
 - pixel_accuracy___background: 0.9772
 - union_over_inter_background: 0.9430
 - dice_coefficient_background: 0.9711
 - pixel_accuracy___snow: 0.9826
 - union_over_inter_snow: 0.9417
 - dice_coefficient_snow: 0.9704
 - pixel_accuracy___clouds: 0.9878
 - union_over_inter_clouds: 0.9509
 - dice_coefficient_clouds: 0.9753
 - pixel_accuracy___water: 1.0002
 - union_over_inter_water: 0.7837
 - dice_coefficient_water: 0.8665
[INFO] saving the model...
```

### Timing Summary:

```text
ModelTrainer                          total [ms]    count [n]        std [ms]       mean [ms]
  +-  __train_epoch:                  121592073.75  60               17121.069      2026534.562     
  +-  __test_epoch:                   27197216.56   60               4201.037       453286.943      
  +-  train:                          148802752.0   1                0.0            148802752.0    
```

Trained on NVIDIA TITAN RTX and an AMD EPYC 7742 64-Core Processor.

:::

::: details Git Commit ([eaa7527](https://github.com/wp99cp/bachelor-thesis/commit/eaa7527)) - 03.05.2023

This was `15775547` slurm job. Trained **without** rooted imbalance weight. Trained on all 36 scenes.

```python
INIT_LR = 0.001  # if using amp the INIT_LR should be below 0.001
MOMENTUM = 0.950
WEIGHT_DECAY = 0.1
NUM_EPOCHS = 256
BATCH_SIZE = 48  # fastest on Euler (assuming Quadro RTX 6000) is 32, 
# however this may be too small (nan loss)
```

:::

::: details Git Commit ([27926e7](https://github.com/wp99cp/bachelor-thesis/commit/27926e7)) - 03.05.2023

This was `15762688` slurm job. Trained on all 36 scenes. The data was normalized using the 30/70 percentile method with
constant values per band (based on the summary stats of the training data). The data was shifted such that the mean is
zero per band. Trained using the rooted imbalance weight.

Snow coverage quite good after the first epoch, water and clouds are still a problem. However, the model is still
learning....

![img.png](../images/results/27926e7/results_after_first_epoch.png)

:::

::: details Git Commit ([8d687e8](https://github.com/wp99cp/bachelor-thesis/commit/8d687e8)) - 30.04.2023

This was `15752952` slurm job. Trained on all 36 scenes. The data was normalized using the 30/70 percentile method with
constant values per band (based on the summary stats of the training data). The data was shifted such that the mean is
zero per band.

This run had some issues with nan loss after 7 epochs. Thus, I increased the batch size from 32 to 48 for `27926e7`.
Additionally, I reduced the imbalance weight by taking its square root for `27926e7`.

![Training Graph](../images/results/8d687e8/training_graph.png)

:::

::: details Git Commit ([1c88144](https://github.com/wp99cp/bachelor-thesis/commit/1c88144)) - 30.04.2023

This was `15685803` slurm job. Trained for 62 epochs, manually stopped. However, trained using only 21 datasets, due to
a missing file. The data was normalized using the `percentile` method, but on a per band and per scene basis. This
approach is not recommended, as it is not consistent.

This run was mainly used to create the summary stats for a later run...

Nevertheless, the model is quite good, but has problems with water.

![Training Graph](../images/results/1c88144/training_graph.png)

:::

## Older Models

Currently by best model is `e40b271`. It works reliably in different scenes and seasons.

For older models the image splitter must be set to legacy mode. This can be done by setting inside the `config.py`.

```python
# [...]
LEGACY_MODE = False  # legacy: True
# [...]
PERCENTILE_CLPPING_DYNAMIC_WORLD_METHOD = True  # legacy: False
# [...]
SIGMA_CLIPPING = False
# [...]
```

::: details Git Commit ([e40b271](https://github.com/wp99cp/bachelor-thesis/commit/e40b271)) - 14.04.2023

This was `14235800` slurm job.

Model trained using amp (mixed precision) and AdamW optimizer. Using all 13 bands and the elevation data.
Great performance except for water.

![Training Graph](../images/results/e40b271/training_graph.png)

:::

::: details Git Commit ([2d0d993](https://github.com/wp99cp/bachelor-thesis/commit/2d0d993)) - 10.04.2023

That was `14035434` slurm job.

Model trained using amp (mixed precision) and AdamW optimizer. Using all 13 bands and the elevation data.

This model has slow convergence, however it is learning quite well.

```
Epoch: 49, train_loss: 0.074335, test_loss: 0.061889
Metrics for validation of epoch:  49
 - pixel_accuracy___background: 0.9733
 - union_over_inter_background: 0.9346
 - dice_coefficient_background: 0.9666
 - pixel_accuracy___snow: 0.9776
 - union_over_inter_snow: 0.9220
 - dice_coefficient_snow: 0.9597
 - pixel_accuracy___clouds: 0.9820
 - union_over_inter_clouds: 0.9302
 - dice_coefficient_clouds: 0.9641
 - pixel_accuracy___water: 0.9996
 - union_over_inter_water: 0.6934
 - dice_coefficient_water: 0.8009
```

:::

::: details Git Commit ([202ea7d](https://github.com/wp99cp/bachelor-thesis/commit/202ea7d)) - 03.04.2023

Trained based on a hand-selected limited dataset, containing of the bands:

```yml
[ '20210116T102351', '20210215T102121', '20210302T101839', '20210401T101559', '20210406T102021', '20210531T101559', '20210625T102021', '20210720T101559', '20210913T102021', '20211018T101939', '20211028T102039', '20211013T101951' ]
```

![inference_7040_5248](../images/results/202ea7d/inference_7040_5248_202ea7d.png)
![inference_9600_7168](../images/results/202ea7d/inference_9600_7168_202ea7d.png)

![Example Image 1](../images/results/202ea7d/example_image_1_202ea7d.png)
![Example Image 2](../images/results/202ea7d/example_image_2_202ea7d.png)

### Problems

Huge Problem on scene `20211023T102101`

- false positive clouds in bright snow
- huge problem with water prediction

![img.png](../images/results/202ea7d/example_image_3.png)

:::

::: details Git Commit ([8ad1ca2](https://github.com/wp99cp/bachelor-thesis/commit/8ad1ca2)) - 03.04.2023

Trained based on a hand-selected limited dataset, containing of the bands:

```yml
[ '20210116T102351', '20210215T102121', '20210302T101839', '20210401T101559', '20210406T102021', '20210531T101559', '20210625T102021', '20210720T101559', '20210913T102021', '20211018T101939', '20211028T102039', '20211013T101951' ]
```

Additionally, the steps per epoch was limited:

```
STEPS_PER_EPOCH = 1024
STEPS_PER_EPOCH_TEST = 128
```

![inference_7040_5248](../images/results/8ad1ca2/inference_7040_5248_8ad1ca2.png)
![inference_9600_7168](../images/results/8ad1ca2/inference_9600_7168_8ad1ca2.png)

![Example Image 1](../images/results/8ad1ca2/example_image_1_8ad1ca2.png)
![Example Image 2](../images/results/8ad1ca2/example_image_2_8ad1ca2.png)

:::

::: details Git Commit ([a2528d2](https://github.com/wp99cp/bachelor-thesis/commit/a2528d2)) - 02.04.2023

This was slurm id `13611024`

Big mode, trained with 243'712 patches from 35 different dates. All 13 channels and the elevation data were used.
Additionally, the elevation data was used.

The training set has the following class distribution:

```
 Total Class Distribution: [0.41002, 0.29181, 0.29019, 0.00798]
```

Results:

![inference_7040_5248](../images/results/a2528d2/inference_7040_5248_a2528d2.png)
![inference_9600_7168](../images/results/a2528d2/inference_9600_7168_a2528d2.png)

![Example Image 1](../images/results/a2528d2/example_image_1_a2528d2.png)
![Example Image 2](../images/results/a2528d2/example_image_2_a2528d2.png)

Compared to the model `202ea7d` , the results for scene `20211023T102101` are much better.

![img.png](../images/results/a2528d2/example_image_3_a2528d2.png)

Winter scene `20211217T102329`:

![Example Image 3](../images/results/a2528d2/example_image_3_original.png)
![Example Image 3](../images/results/a2528d2/example_image_3_training_mask.png)
![Example Image 3](../images/results/a2528d2/example_image_3_original_with_training_mask.png)
![Example Image 3](../images/results/a2528d2/example_image_3_predicted_classification.png)
![Example Image 3](../images/results/a2528d2/example_image_3_original_with_predicted_classification.png)
![Example Image 3](../images/results/a2528d2/example_image_3_both_masks.png)

:::

::: details Git Commit ([8fe0cea](https://github.com/wp99cp/bachelor-thesis/commit/8fe0cea)) - 02.04.2023

This was slurm id `13611087`.

Big model, trained with 243'712 patches from 35 different dates. All 13 channels and no additional auxiliary data were
used. The training set has the following class distribution:

```
 Total Class Distribution: [0.41002, 0.29181, 0.29019, 0.00798]
```

Results:

![inference_7040_5248](../images/results/8fe0cea/inference_7040_5248_8fe0cea.png)
![inference_9600_7168](../images/results/8fe0cea/inference_9600_7168_8fe0cea.png)

![Example Image 1](../images/results/8fe0cea/example_image_1_8fe0cea.png)
![Example Image 2](../images/results/8fe0cea/example_image_2_8fe0cea.png)

:::

::: details Git Commit ([4c654a2](https://github.com/wp99cp/bachelor-thesis/commit/4c654a2)) - 29.03.2023

Trained using auto masks for all images (73) scenes, using 600 samples per date. The training set has the following
class distribution:

```
 Total Class Distribution: [0.27588 0.23729 0.48049 0.00634 0.     ]
« Background, Snow, Clouds, Water, Semi-Transparent Clouds
```

Configuration:

```
 - SAMPLES_PER_DATE = 600
 - IMG_SIZE = 256
 - NUM_ENCODED_CHANNELS = 5
 - SELECTED_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
 - DATA_DIR = /scratch/tmp.13277407.pucyril/data
 - EXTRACTED_RAW_DATA = /cluster/scratch/pucyril/data_sources/extracted_raw_data
 - MAKS_PATH = /cluster/scratch/pucyril/data_sources/masks
 - DATASET_DIR = /scratch/tmp.13277407.pucyril/data/dataset
 - RESULTS = /scratch/tmp.13277407.pucyril/results
 - LIMIT_DATES = 0

Found 73 dates
```

The model was trained for 7 epochs with a batch size of 16 and image size of 256. The architecture is as follows:

```
-----------------------------------------------------------------------------
         Layer (type)            Input Shape         Param #     Tr. Param #
=============================================================================
             Conv2d-1      [1, 13, 256, 256]           7,488           7,488
        BatchNorm2d-2      [1, 64, 256, 256]             128             128
               ReLU-3      [1, 64, 256, 256]               0               0
             Conv2d-4      [1, 64, 256, 256]          36,864          36,864
        BatchNorm2d-5      [1, 64, 256, 256]             128             128
               ReLU-6      [1, 64, 256, 256]               0               0
          MaxPool2d-7      [1, 64, 256, 256]               0               0
         DoubleConv-8      [1, 64, 128, 128]         221,696         221,696
          MaxPool2d-9     [1, 128, 128, 128]               0               0
        DoubleConv-10       [1, 128, 64, 64]         885,760         885,760
         MaxPool2d-11       [1, 256, 64, 64]               0               0
        DoubleConv-12       [1, 256, 32, 32]       3,540,992       3,540,992
         MaxPool2d-13       [1, 512, 32, 32]               0               0
        DoubleConv-14       [1, 512, 16, 16]      14,159,872      14,159,872
   ConvTranspose2d-15      [1, 1024, 16, 16]       2,097,664       2,097,664
        DoubleConv-16      [1, 1024, 32, 32]       7,079,936       7,079,936
   ConvTranspose2d-17       [1, 512, 32, 32]         524,544         524,544
        DoubleConv-18       [1, 512, 64, 64]       1,770,496       1,770,496
   ConvTranspose2d-19       [1, 256, 64, 64]         131,200         131,200
        DoubleConv-20     [1, 256, 128, 128]         442,880         442,880
   ConvTranspose2d-21     [1, 128, 128, 128]          32,832          32,832
        DoubleConv-22     [1, 128, 256, 256]         110,848         110,848
            Conv2d-23      [1, 64, 256, 256]             260             260
=============================================================================
Total params: 31,043,588
Trainable params: 31,043,588
Non-trainable params: 0
-----------------------------------------------------------------------------
```

![Inference results](../images/results/4c654a2/inference_2021-04-21_260.png)

:::

::: details Git Commit ([913b2a3](https://github.com/wp99cp/bachelor-thesis/tree/913b2a38017)) - 14.03.2023

Trained on 10_240 of a single scene (20211008T101829) using all 13 bands.

The training set has the following class distribution:

```
CLASS_WEIGHTS = [0.25052, 0.00214, 0.01381, 0.02479]
```

The model was trained for 25 epochs with a batch size of 64. The architecture is as follows:

```
-----------------------------------------------------------------------------
         Layer (type)            Input Shape         Param #     Tr. Param #
=============================================================================
             Conv2d-1      [1, 13, 128, 128]           7,488           7,488
        BatchNorm2d-2      [1, 64, 128, 128]             128             128
               ReLU-3      [1, 64, 128, 128]               0               0
             Conv2d-4      [1, 64, 128, 128]          36,864          36,864
        BatchNorm2d-5      [1, 64, 128, 128]             128             128
               ReLU-6      [1, 64, 128, 128]               0               0
          MaxPool2d-7      [1, 64, 128, 128]               0               0
         DoubleConv-8        [1, 64, 64, 64]         221,696         221,696
          MaxPool2d-9       [1, 128, 64, 64]               0               0
        DoubleConv-10       [1, 128, 32, 32]         885,760         885,760
         MaxPool2d-11       [1, 256, 32, 32]               0               0
        DoubleConv-12       [1, 256, 16, 16]       3,540,992       3,540,992
         MaxPool2d-13       [1, 512, 16, 16]               0               0
        DoubleConv-14         [1, 512, 8, 8]      14,159,872      14,159,872
   ConvTranspose2d-15        [1, 1024, 8, 8]       2,097,664       2,097,664
        DoubleConv-16      [1, 1024, 16, 16]       7,079,936       7,079,936
   ConvTranspose2d-17       [1, 512, 16, 16]         524,544         524,544
        DoubleConv-18       [1, 512, 32, 32]       1,770,496       1,770,496
   ConvTranspose2d-19       [1, 256, 32, 32]         131,200         131,200
        DoubleConv-20       [1, 256, 64, 64]         442,880         442,880
   ConvTranspose2d-21       [1, 128, 64, 64]          32,832          32,832
        DoubleConv-22     [1, 128, 128, 128]         110,848         110,848
            Conv2d-23      [1, 64, 128, 128]             260             260
=============================================================================
Total params: 31,043,588
Trainable params: 31,043,588
Non-trainable params: 0
-----------------------------------------------------------------------------
```

Prediction on a single image (I selected a particular accurate result, in general the results are not that good):

![Inference results](../images/results/913b2a3801/inference_2021-10-08_2374.png)

:::

