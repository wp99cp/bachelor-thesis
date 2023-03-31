import os

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from configs.config import IMAGE_SIZE, MASK_DATASET_PATH, DEVICE, BASE_OUTPUT, NUM_ENCODED_CHANNELS, THRESHOLD


def make_predictions(model, image_path):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = np.load(image_path)
        # image = image['arr_0'] # this is only needed for the npy files saved with np.savez_compressed
        orig = image.copy()
        # orig = np.moveaxis(orig, 0, -1)

        # find the filename and generate the path to ground truth
        # mask
        filename = image_path.split(os.path.sep)[-1]
        # replace the extension of the image with .png
        filename = filename.replace(".npy", ".png")
        ground_truth_path = os.path.join(MASK_DATASET_PATH, filename)

        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gt_mask = cv2.imread(ground_truth_path, cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.resize(gt_mask, (IMAGE_SIZE, IMAGE_SIZE))

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.moveaxis(image, 2, 0)
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(DEVICE)

        # make the prediction, pass the results through the softmax
        # function, and convert the result to a NumPy array
        _, pred_mask = model(image)
        pred_mask = pred_mask.squeeze()  # squeeze removes the batch dimension
        pred_mask = pred_mask.cpu().numpy()  # move to CPU and convert to numpy array

        # set pixels with a value greater than 0.5 to 1, and set
        # pixels with a value less than or equal to 0.5 to 0
        # pred_mask[pred_mask > THRESHOLD] = 1
        # pred_mask[pred_mask <= THRESHOLD] = 0

        # get filename from path
        filename = image_path.split(os.path.sep)[-1]
        filename = filename.replace(".npy", "")

        # prepare a plot for visualization
        print_results(orig, gt_mask, pred_mask, filename)


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


def print_results(orig_image, orig_mask, predMask, imagePath: str):
    # initialize our figure
    matplotlib.use('Agg')
    figure, ax = plt.subplots(nrows=2, ncols=6, figsize=(30, 10))

    # create a legend for the mask

    # plot the original image, its mask, and the predicted mask
    rgb = orig_image[:, :, 1:4]
    color_infrared = orig_image[:, :, [2, 3, 7]]
    short_wave_infrared = orig_image[:, :, [12, 8, 3]]

    orig_mask = orig_mask.astype(int)

    # Create a mask of the non-white pixels in the mask image
    # the
    mask_alpha = np.zeros(orig_mask.shape, dtype=np.uint8)
    mask_alpha[orig_mask != 0] = 1

    orig_mask = (orig_mask * NUM_ENCODED_CHANNELS) / 255
    cmap, rgb_mask = colorize_mask(orig_mask)

    # normalize image to [0, 1]
    rgb = rgb.astype(np.float32)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
    color_infrared = color_infrared.astype(np.float32)
    color_infrared = (color_infrared - np.min(color_infrared)) / (np.max(color_infrared) - np.min(color_infrared))
    short_wave_infrared = short_wave_infrared.astype(np.float32)
    short_wave_infrared = (short_wave_infrared - np.min(short_wave_infrared)) / (
            np.max(short_wave_infrared) - np.min(short_wave_infrared))

    rgb_increased_gamma = np.power(rgb, 1 / 2.2)

    ax[0, 0].imshow(rgb)

    # Combine rgb and rgb_mask into a single image using rgb as background and rgb_mask as foreground with alpha mask
    mask_alpha = np.stack((mask_alpha,) * 3, axis=-1)
    blended_image = rgb * (1 - mask_alpha) + rgb_mask * mask_alpha

    ax[0, 1].imshow(blended_image)

    ax[0, 2].imshow(color_infrared)
    ax[0, 3].imshow(short_wave_infrared)
    ax[0, 4].imshow(rgb_increased_gamma)

    # set the titles of the subplots
    ax[0, 0].set_title("Original Image")
    ax[0, 1].set_title("Original Mask")
    ax[0, 2].set_title("Color Infrared")
    ax[0, 3].set_title("Short Wave Infrared")
    ax[0, 4].set_title("RGB Increased Gamma")

    ax[1, 2].set_title("Predicted Background")
    ax[1, 3].set_title("Predicted Snow")
    ax[1, 4].set_title("Predicted Cloud")
    ax[1, 5].set_title("Predicted Water")
    ax[1, 0].set_title("Predicted Mask")
    ax[1, 1].set_title("Diff Training - Prediction")

    for i in [2, 3, 4, 5]:
        figure.colorbar(ax[1, i].imshow(predMask[i - 2, :, :], vmin=0, vmax=1), ax=ax[1, i])
        ax[1, i].text(12.5, 300, f"Min: {np.min(predMask[i - 2, :, :]):.2f}, "
                                 f"Max: {np.max(predMask[i - 2, :, :]):.2f}, "
                                 f"Mean: {np.mean(predMask[i - 2, :, :]):.2f}")

    # add legend to figure 1
    legend_elements = [
        Patch(facecolor=cmap[0] / 255.0, label='Background'),
        Patch(facecolor=cmap[1] / 255.0, label='Snow'),
        Patch(facecolor=cmap[2] / 255.0, label='Clouds'),
        Patch(facecolor=cmap[3] / 255.0, label='Water'),
        # Patch(facecolor=cmap[4] / 255.0, label='Semi-Transparent Cloud')
    ]

    ax[0, 1].legend(handles=legend_elements, loc='upper right')

    predMask = predMask.transpose(1, 2, 0)
    predMask[:, :, 1] = (predMask[:, :, 1] > THRESHOLD).astype(int)
    predMask[:, :, 2] = (predMask[:, :, 2] > THRESHOLD).astype(int)
    predMask[:, :, 3] = (predMask[:, :, 3] > THRESHOLD).astype(int)

    # map to (255, 255) by setting ones of the layers to 1, 2, 3
    pred_mask_encoded = np.zeros((predMask.shape[0], predMask.shape[1]), dtype=np.uint8)
    pred_mask_encoded[predMask[:, :, 1] == 1] = 1
    pred_mask_encoded[predMask[:, :, 2] == 1] = 2
    pred_mask_encoded[predMask[:, :, 3] == 1] = 3

    # compute difference between predicted mask and original mask
    diff_mask = pred_mask_encoded - orig_mask
    diff_mask[diff_mask != 0] = 1

    # plot the difference mask
    figure.colorbar(ax[1, 1].imshow(diff_mask, cmap='bwr', vmin=0, vmax=1), ax=ax[1, 1])

    cmap, rgb_pred_mask = colorize_mask(pred_mask_encoded)
    ax[1, 0].imshow(rgb_pred_mask)
    ax[1, 0].legend(handles=legend_elements, loc='upper right')

    # set the layout of the figure and display it
    figure.tight_layout()

    inference_path = os.path.join(BASE_OUTPUT, f"inference_{imagePath}.png")
    figure.savefig(inference_path)
