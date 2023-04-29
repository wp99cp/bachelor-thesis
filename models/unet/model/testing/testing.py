# import the necessary packages form the pre-processing/image_splitter
import os
import sys

import numpy as np
import rasterio

from SentinelDataLoader import SentinelDataLoader
from configs.config import BASE_OUTPUT, DATASET_PATH
from utils.rasterio_helpers import get_profile

sys.path.insert(0, os.environ['BASE_DIR'] + '/helper-scripts/python_helpers')
# noinspection PyUnresolvedReferences
from pipeline_config import load_pipeline_config, get_dates


def run_testing(pipeline_config, model_file_name='unet'):
    print("[INFO] running testing...")

    inference_dates = get_dates(pipeline_config, pipeline_step="inference")
    dates = get_dates(pipeline_config, pipeline_step="testing")
    print(f"[INFO] Run inference on dates: {dates}")
    assert any(elem in inference_dates for elem in
               dates), "Testing is only possible if inference is enabled for the same dates."

    for s2_date in dates:
        model_name = model_file_name.split(".")[0]
        path_prefix = os.path.join(f"{s2_date}_pred", model_name)

        run_testing_on_date(s2_date, path_prefix)


def run_testing_on_date(s2_date, path_prefix):
    dataloader = SentinelDataLoader(dates=[s2_date])
    dataloader.open_date(s2_date, fast_mode=True)

    # Step 1: load predicted, ground truth and the mask used for training
    with rasterio.open(os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_mask_prediction.jp2")) as mask:
        prediction = mask.read(1)

    with rasterio.open(os.path.join(DATASET_PATH, '..', 'ground_truth_masks', s2_date, "mask.jp2")) as mask:
        ground_truth = mask.read(1)

        # merge classes for dense and sparse clouds, set both to class 2 (dense clouds)
        ground_truth[ground_truth == 4] = 2

    training_mask = dataloader.get_mask(s2_date)

    # Step 3: calculate metrics
    print("\n\n\n=================\nTesting results:\n=================\n")

    print("Pixel counts (Prediction, before dropping):")
    report_pixel_counts(prediction)

    # We have the following problem, that the training mask and ground truth mask
    # may not be computed for the same areas as the prediction mask (which spans the whole image).
    # During evaluation, we only consider the areas where all three masks are defined.
    # As our metrics are geometrically invariant, we can simply ignore the areas
    # where the masks are not defined by dropping the corresponding pixels in all three masks.

    # load coverage masks
    prediction_coverage = np.ones(prediction.shape, dtype=np.bool_)
    with rasterio.open(os.path.join(DATASET_PATH, '..', 'ground_truth_masks', s2_date, "mask_coverage.jp2")) as mask:
        ground_truth_coverage = mask.read(1)
    training_mask_coverage = dataloader.get_mask_coverage(s2_date)

    __create_different_mask(ground_truth, prediction, ground_truth_coverage, s2_date, path_prefix,
                            name="mask_diff_ground_truth")
    __create_different_mask(training_mask, prediction, training_mask_coverage, s2_date, path_prefix,
                            name="mask_diff_training")

    # intersect coverage masks
    coverage = np.logical_and(ground_truth_coverage, training_mask_coverage, prediction_coverage)
    coverage = coverage.flatten()

    print(f"Coverage: {((np.sum(coverage) / coverage.shape[0]) * 100):02.3f} %")
    print(f"Dropped pixels: {coverage.shape[0] - np.sum(coverage)}")
    print()

    # reduce to 1D arrays
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()
    training_mask = training_mask.flatten()

    # drop pixels where any of the masks is undefined (i.g. coverage = 0)
    prediction = prediction[coverage]
    ground_truth = ground_truth[coverage]
    training_mask = training_mask[coverage]

    print("Pixel counts (Prediction, after dropping):")
    report_pixel_counts(prediction)

    print("Pixel counts (Ground truth):")
    report_pixel_counts(ground_truth)

    report_accuracies(ground_truth, prediction, training_mask)
    report_uio(ground_truth, prediction)
    report_dice(ground_truth, prediction)
    report_jaccard(ground_truth, prediction)

    print("\n\n\n=================\nTesting results:\n=================\n")


def report_jaccard(ground_truth, prediction):
    def __calculate_jaccard(ground_truth, prediction):
        intersection = np.sum(ground_truth * prediction)
        union = np.sum(ground_truth) + np.sum(prediction) - intersection
        return intersection / union

    print("Class Jaccard (against ground_truth):")

    snow_jaccard = __calculate_jaccard(ground_truth == 1, prediction == 1)
    print(f"- Snow:\t\t{snow_jaccard:02.3f}")

    cloud_jaccard = __calculate_jaccard(ground_truth == 2, prediction == 2)
    print(f"- Cloud:\t{cloud_jaccard:02.3f}")

    water_jaccard = __calculate_jaccard(ground_truth == 3, prediction == 3)
    print(f"- Water:\t{water_jaccard:02.3f}")

    background_jaccard = __calculate_jaccard(ground_truth == 0, prediction == 0)
    print(f"- Background:\t{background_jaccard:02.3f}")

    print("")
    print(f"\nMean Jaccard (without background):\t{(snow_jaccard + cloud_jaccard + water_jaccard) / 3:02.3f}")
    print("")


def report_dice(ground_truth, prediction):
    def __calculate_dice(ground_truth, prediction):
        intersection = np.sum(ground_truth * prediction)
        union = np.sum(ground_truth) + np.sum(prediction)
        return 2 * intersection / union

    print("Class Dice (against ground_truth):")

    snow_dice = __calculate_dice(ground_truth == 1, prediction == 1)
    print(f"- Snow:\t\t{snow_dice:02.3f}")

    cloud_dice = __calculate_dice(ground_truth == 2, prediction == 2)
    print(f"- Cloud:\t{cloud_dice:02.3f}")

    water_dice = __calculate_dice(ground_truth == 3, prediction == 3)
    print(f"- Water:\t{water_dice:02.3f}")

    background_dice = __calculate_dice(ground_truth == 0, prediction == 0)
    print(f"- Background:\t{background_dice:02.3f}")

    print("")
    print(f"Mean Dice (without background):\t{(snow_dice + cloud_dice + water_dice) / 3:02.3f}")
    print("")


def report_uio(ground_truth, prediction):
    def __calculate_iou(A, B):
        intersection = np.logical_and(A, B)
        union = np.logical_or(A, B)
        return np.sum(intersection) / np.sum(union)

    print("Class IOU (against ground_truth):")
    snow_iou = __calculate_iou(ground_truth == 1, prediction == 1)
    print(f"- Snow:\t\t{snow_iou:02.3f}")

    cloud_iou = __calculate_iou(ground_truth == 2, prediction == 2)
    print(f"- Cloud:\t{cloud_iou:02.3f}")

    water_iou = __calculate_iou(ground_truth == 3, prediction == 3)
    print(f"- Water:\t{water_iou:02.3f}")

    background_iou = __calculate_iou(ground_truth == 0, prediction == 0)
    print(f"- Background:\t{background_iou:02.3f}")

    print("")
    print(f"Mean IOU (without background):\t{(snow_iou + cloud_iou + water_iou) / 3:02.3f}")
    print("")


def report_pixel_counts(mask):
    # mask may be a 2D array or a 1D array
    number_of_pixels = mask.size if mask.ndim == 1 else mask.shape[0] * mask.shape[1]

    print(f"Number of pixels:\t\t{number_of_pixels:10d} ( 100.000 % )")

    numer_of_snow_pixels = np.sum(mask == 1)
    print(f"- Number of snow pixels:\t{numer_of_snow_pixels:10d} "
          f"( {(numer_of_snow_pixels / number_of_pixels * 100):02.3f} % )")

    numer_of_cloud_pixels = np.sum(mask == 2)
    print(f"- Number of cloud pixels:\t{numer_of_cloud_pixels:10d} "
          f"( {(numer_of_cloud_pixels / number_of_pixels * 100):02.3f} % )")

    numer_of_water_pixels = np.sum(mask == 3)
    print(f"- Number of water pixels:\t{numer_of_water_pixels:10d} "
          f"( {(numer_of_water_pixels / number_of_pixels * 100):02.3f} % )")

    numer_of_background_pixels = np.sum(mask == 0)
    print(f"- Number of background pixels:\t{numer_of_background_pixels:10d} "
          f"( {(numer_of_background_pixels / number_of_pixels * 100):02.3f} % )")

    print("")


def report_accuracies(ground_truth, prediction, training_mask):
    number_of_pixels = prediction.size

    print("Accuracy:")
    accuracy = np.sum(prediction == training_mask) / number_of_pixels
    print(f"- Accuracy (against training_mask): {accuracy}")

    accuracy = np.sum(prediction == ground_truth) / number_of_pixels
    print(f"- Accuracy (against ground_truth): {accuracy}")

    print("")
    print("Class accuracies (against ground_truth):")

    # snow is class 1
    snow_accuracy = np.sum((prediction == 1) & (ground_truth == 1)) / np.sum(ground_truth == 1)
    print(f"- Snow: {snow_accuracy}")

    # water is class 2
    cloud_accuracy = np.sum((prediction == 2) & (ground_truth == 2)) / np.sum(ground_truth == 2)
    print(f"- Cloud: {cloud_accuracy}")

    # water is class 3
    water_accuracy = np.sum((prediction == 3) & (ground_truth == 3)) / np.sum(ground_truth == 3)
    print(f"- Water: {water_accuracy}")

    # background is class 0
    background_accuracy = np.sum((prediction == 0) & (ground_truth == 0)) / np.sum(ground_truth == 0)
    print(f"- Background: {background_accuracy}")

    print("")


def __create_different_mask(ground_truth, prediction, coverage, s2_date, path_prefix, name="mask_diff"):
    # calculate the difference between the predicted mask and the training mask
    # and save it as a jp2 file

    # calculate the difference (pixel wise)
    # define a lookup table for the output values
    # calculate the difference using vectorized operations and the lookup table
    lookup = np.array([0, 11, 12, 13, 1, 0, 14, 15, 2, 4, 0, 16, 3, 5, 6, 0])
    diff = lookup[ground_truth * 4 + prediction]
    diff = diff.astype('int8')

    # set the difference to -1 if there is no coverage
    diff[coverage == 0] = -1  # set the difference to -1 if there is no coverage

    # save the difference
    path = os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_{name}.jp2")

    profile = get_profile(s2_date)
    with rasterio.open(path, 'w', **profile) as mask_file:
        mask_file.write(diff, 1)
