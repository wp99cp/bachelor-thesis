import torch
from torch import Tensor

from configs.config import THRESHOLD, NUM_CLASSES, CLASS_NAMES


# pixel accuracy
# defined as the percentage of correctly classified pixels
# pixl accuracy suffers from class imbalance
def pixel_accuracy(y_head, y_true, class_index=0):
    """
    Calculates the pixel accuracy for a given class.
    defined as the percentage of correctly classified pixels

    pixl accuracy suffers from class imbalance

    :param y_head: the predicted mask for all classes
    :param y_true: the ground truth mask for all classes
    :param class_index: the index of the class we are interested in

    :return: the pixel accuracy
    """

    y_head_class = y_head[:, class_index, :, :]
    y_true_class = y_true[:, class_index, :, :]

    # apply a threshold to the predicted mask
    # this converts the mask to a binary mask
    y_head_class = torch.where(y_head_class > THRESHOLD, 1, 0)

    correct_pixels = torch.sum(y_head_class == y_true_class)
    total_pixels = y_head_class.shape[0] * y_head_class.shape[1] * y_head_class.shape[2]
    return correct_pixels / total_pixels


def UoI(y_head, y_true, class_index=0):
    # intersection over union (IoU)
    # defined as the ratio between the number of pixels that are correctly
    # classified and the total number of pixels that should have been
    # correctly classified

    y_head_class = y_head[:, class_index, :, :]
    y_true_class = y_true[:, class_index, :, :]

    # apply a threshold to the predicted mask
    # this converts the mask to a binary mask
    y_head_class = torch.where(y_head_class > THRESHOLD, 1, 0)

    intersection = torch.sum(y_head_class * y_true_class)
    union = torch.sum(y_head_class) + torch.sum(y_true_class) - intersection

    return intersection / union


def dice_coefficient(y_head, y_true, class_index=0):
    # dice coefficient (F1 score)
    # defined as the ratio between the number of pixels that are correctly
    # classified and the total number of pixels that should have been
    # correctly classified

    y_head_class = y_head[:, class_index, :, :]
    y_true_class = y_true[:, class_index, :, :]

    # apply a threshold to the predicted mask
    # this converts the mask to a binary mask
    y_head_class = torch.where(y_head_class > THRESHOLD, 1, 0)

    intersection = torch.sum(y_head_class * y_true_class)
    union = torch.sum(y_head_class) + torch.sum(y_true_class)

    return 2 * intersection / union


def get_segmentation_metrics():
    """

    Initialize additional metrics
    for that we are considering the following metrics:
     - pixel accuracy
     - intersection over union (IoU)
     - dice coefficient (F1 score)

    :return: a list of metrics functions
    """

    metrics = []

    # add metrics for every class
    for i in range(NUM_CLASSES):
        def _pixel_accuracy_class(y_head, y_true, cidx=i):
            return pixel_accuracy(y_head, y_true, class_index=cidx)

        _pixel_accuracy_class.__name__ = f"pixel_accuracy___{CLASS_NAMES[i]}"
        metrics.append(_pixel_accuracy_class)

        def _UoI_class(y_head, y_true, cidx=i):
            return UoI(y_head, y_true, class_index=cidx)

        _UoI_class.__name__ = f"union_over_inter_{CLASS_NAMES[i]}"
        metrics.append(_UoI_class)

        def _dice_coefficient_class(y_head, y_true, cidx=i):
            return dice_coefficient(y_head, y_true, class_index=cidx)

        _dice_coefficient_class.__name__ = f"dice_coefficient_{CLASS_NAMES[i]}"
        metrics.append(_dice_coefficient_class)

    return metrics
