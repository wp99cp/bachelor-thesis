import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from configs.config import IMAGE_SIZE, MASK_DATASET_PATH, DEVICE, THRESHOLD, BASE_OUTPUT


def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        orig = image.copy()

        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(MASK_DATASET_PATH, filename)

        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath, cv2.IMREAD_UNCHANGED)
        gtMask = cv2.resize(gtMask, (IMAGE_SIZE, IMAGE_SIZE))

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        predMask = predMask

        # set pixels with a value greater than 0.5 to 1, and set
        # pixels with a value less than or equal to 0.5 to 0
        predMask[predMask > THRESHOLD] = 1
        predMask[predMask <= THRESHOLD] = 0

        # prepare a plot for visualization
        print_results(orig, gtMask, predMask)


def print_results(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))

    # create a legend for the mask

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)

    ax[2].imshow(predMask[0, :, :])
    ax[3].imshow(predMask[1, :, :])
    ax[4].imshow(predMask[2, :, :])
    ax[5].imshow(predMask[3, :, :])

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")

    ax[2].set_title("Predicted Background")
    ax[3].set_title("Predicted Snow")
    ax[4].set_title("Predicted Cloud")
    ax[5].set_title("Predicted Water")

    # set the layout of the figure and display it
    figure.tight_layout()

    inference_path = os.path.join(BASE_OUTPUT, "inference.png")
    figure.savefig(inference_path)
