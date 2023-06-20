import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from src.models.unet.configs.config import NUM_ENCODED_CHANNELS, IMAGE_SIZE, BASE_OUTPUT


def __colorize_mask(mask_):
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


def print_results(orig_image, orig_mask, predMask, coords, file_path_prefix=''):
    """
    Print the results of a single patch prediction.
    :param orig_image: the original image
    :param orig_mask: the original mask
    :param predMask: the predicted mask
    :param coords: the coordinates of the patch
    :param file_path_prefix: the prefix of the file path
    """

    # initialize our figure
    matplotlib.use('Agg')
    figure, ax = plt.subplots(nrows=2, ncols=6, figsize=(30, 10))

    (x, y) = coords

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
    cmap, rgb_mask = __colorize_mask(orig_mask)

    # normalize image to [0, 1]
    rgb = rgb.astype(np.float32)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
    color_infrared = color_infrared.astype(np.float32)
    color_infrared = (color_infrared - np.min(color_infrared)) / (np.max(color_infrared) - np.min(color_infrared))
    short_wave_infrared = short_wave_infrared.astype(np.float32)
    short_wave_infrared = (short_wave_infrared - np.min(short_wave_infrared)) / (
            np.max(short_wave_infrared) - np.min(short_wave_infrared))

    rgb_increased_gamma = np.power(rgb, 1 / 2.2)

    # print the location of the patch within the tile
    # mark the location with a red rectangle in the original tile (10980x10980)
    rect = matplotlib.patches.Rectangle((y, x), IMAGE_SIZE, IMAGE_SIZE, facecolor='r')
    ax[0, 5].add_patch(rect)
    ax[0, 5].set_xlim(0, 10980)
    ax[0, 5].set_ylim(10980, 0)

    ax[0, 0].imshow(rgb)

    # Combine rgb and rgb_mask into a single image using rgb as background and rgb_mask as foreground with alpha mask
    mask_alpha = np.stack((mask_alpha,) * 3, axis=-1)
    blended_image = rgb_increased_gamma * (1 - mask_alpha) + rgb_mask * mask_alpha
    blended_image = np.clip(blended_image, 0, 1)

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
    ax[0, 5].set_title("Position within Tile")

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
    pred_mask_encoded = np.argmax(predMask, axis=2)

    # compute difference between predicted mask and original mask
    diff_mask = pred_mask_encoded - orig_mask
    diff_mask[diff_mask != 0] = 1

    # plot the difference mask
    figure.colorbar(ax[1, 1].imshow(diff_mask, cmap='bwr', vmin=0, vmax=1), ax=ax[1, 1])

    mask_alpha = np.zeros(mask_alpha.shape, dtype=np.uint8)
    mask_alpha[pred_mask_encoded != 0] = 1
    cmap, rgb_pred_mask = __colorize_mask(pred_mask_encoded)

    blended_image_prediction = rgb_increased_gamma * (1 - mask_alpha) + rgb_pred_mask * mask_alpha
    blended_image_prediction = np.clip(blended_image_prediction, 0, 1)

    ax[1, 0].imshow(blended_image_prediction)
    ax[1, 0].legend(handles=legend_elements, loc='upper right')

    # set the layout of the figure and display it
    figure.tight_layout()

    inference_path = os.path.join(BASE_OUTPUT, file_path_prefix, f"inference_{x}_{y}.png")
    figure.savefig(inference_path)

    plt.close(figure)
