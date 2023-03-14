# Unet

Resources for Python Code

- [milesial/Pytorch-UNet/](https://github.com/milesial/Pytorch-UNet/tree/master)
- [U-Net: Training Image Segmentation Models in PyTorch](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)

## Results of Previous Iterations

::: details Git Commit ([913b2a3801](https://github.com/wp99cp/bachelor-thesis/tree/913b2a38017)) - 14.03.2023

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