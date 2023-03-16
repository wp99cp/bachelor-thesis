import os

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pytorch_model_summary import summary

from configs.config import IMAGE_SIZE, MASK_DATASET_PATH, DEVICE, BASE_OUTPUT, NUM_CHANNELS, NUM_ENCODED_CHANNELS


def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()

    print(summary(model, torch.zeros((1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE),
                  show_input=True,
                  max_depth=2))

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = np.load(imagePath)
        image = image['arr_0']
        image = image.astype(np.float32)
        image = image / 255.0
        orig = image.copy()
        orig = np.moveaxis(orig, 0, -1)

        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        # replace the extension of the image with .png
        filename = filename.replace(".npz", ".png")
        groundTruthPath = os.path.join(MASK_DATASET_PATH, filename)

        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath, cv2.IMREAD_UNCHANGED)
        gtMask = cv2.resize(gtMask, (IMAGE_SIZE, IMAGE_SIZE))

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(DEVICE)

        # make the prediction, pass the results through the softmax
        # function, and convert the result to a NumPy array
        _, predMask = model(image)
        predMask = predMask.squeeze()  # squeeze removes the batch dimension
        predMask = predMask.cpu().numpy()  # move to CPU and convert to numpy array

        # set pixels with a value greater than 0.5 to 1, and set
        # pixels with a value less than or equal to 0.5 to 0
        # predMask[predMask > THRESHOLD] = 1
        # predMask[predMask <= THRESHOLD] = 0

        # get filename from path
        filename = imagePath.split(os.path.sep)[-1]
        filename = filename.replace(".npz", "")

        # prepare a plot for visualization
        print_results(orig, gtMask, predMask, filename)


def colorize_mask(mask_):
    # Create color map with 4 colors
    cmap_ = np.array([
        [255, 255, 255],  # background
        [0, 0, 255],  # snow is blue
        [0, 255, 0],  # clouds are green
        [255, 0, 0],  # water is red
        [0, 100, 0],  # semi-transparent clouds are dark green
        [0, 0, 0]  # unknown is black
    ])

    # convert to scalar type
    mask_ = np.clip(mask_, 0, NUM_ENCODED_CHANNELS)
    mask_ = mask_.astype(int)
    mask_ = cmap_[mask_]
    mask_ = mask_.astype(np.uint8)
    return cmap_, mask_


def print_results(origImage, origMask, predMask, imagePath: str):
    # initialize our figure
    matplotlib.use('Agg')
    figure, ax = plt.subplots(nrows=1, ncols=6, figsize=(25, 5))

    # create a legend for the mask

    # plot the original image, its mask, and the predicted mask
    rgb = origImage[:, :, 1:4]
    origMask = origMask.astype(int)
    origMask = (origMask * NUM_ENCODED_CHANNELS) / 255
    cmap, rgb_mask = colorize_mask(origMask)

    ax[0].imshow(rgb)
    ax[1].imshow(rgb_mask)

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")

    # ax[2].set_title("Predicted Background")
    ax[2].set_title("Predicted Snow")
    ax[3].set_title("Predicted Cloud")
    ax[4].set_title("Predicted Water")
    ax[5].set_title("Predicted Semi-Transparent Cloud")

    # add colorbar to figures 2 - 5
    figure.colorbar(ax[2].imshow(predMask[0, :, :], vmin=0, vmax=1), ax=ax[2])
    figure.colorbar(ax[3].imshow(predMask[1, :, :], vmin=0, vmax=1), ax=ax[3])
    figure.colorbar(ax[4].imshow(predMask[2, :, :], vmin=0, vmax=1), ax=ax[4])
    figure.colorbar(ax[5].imshow(predMask[3, :, :], vmin=0, vmax=1), ax=ax[5])

    # add legend to figure 1
    legend_elements = [
        Patch(facecolor=cmap[0] / 255.0, label='Background'),
        Patch(facecolor=cmap[1] / 255.0, label='Snow'),
        Patch(facecolor=cmap[2] / 255.0, label='Clouds'),
        Patch(facecolor=cmap[3] / 255.0, label='Water'),
        Patch(facecolor=cmap[4] / 255.0, label='Semi-Transparent Cloud')
    ]

    ax[1].legend(handles=legend_elements, loc='upper right')

    # set the layout of the figure and display it
    figure.tight_layout()

    inference_path = os.path.join(BASE_OUTPUT, f"inference_{imagePath}.png")
    figure.savefig(inference_path)
