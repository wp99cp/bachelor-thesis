# import the necessary packages form the pre-processing/dataset_creator
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import rasterio
from matplotlib import pyplot as plt

from configs.config import BASE_OUTPUT, BASE_DIR
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

    results = []

    if pipeline_config["testing"]["use_cache"] == 0:
        for s2_date in dates:
            model_name = model_file_name.split(".")[0]
            path_prefix = os.path.join(f"{os.environ['TILE_NAME']}_{s2_date}_pred", model_name)

            result = run_testing_on_date(s2_date, path_prefix)
            if result["mean_iou"] is None:
                raise Exception("Mean IoU is None")

            results.append({
                'date': s2_date[0:4] + "-" + s2_date[4:6] + "-" + s2_date[6:8],
                'mean_iou': result["mean_iou"]
            })

        print(f"\n\n\n====================\n====================\n====================\n\n\n")
        print(results)

    # sort results by date
    results = sorted(results, key=lambda k: k['date'])

    # save results to file
    if pipeline_config["testing"]["use_cache"] == 0:
        np.save(os.path.join(BASE_OUTPUT, f"difference_to_exolabs_{os.environ['TILE_NAME']}.npy"), results)

    # load results from file
    if pipeline_config["testing"]["use_cache"] == 1:
        results = np.load(os.path.join(BASE_OUTPUT, f"difference_to_exolabs_{os.environ['TILE_NAME']}.npy"),
                          allow_pickle=True)

    labels = [x['date'] for x in results]
    labels = [datetime.strptime(d, '%Y-%m-%d') for d in labels]

    mean_iou_cloud = [x['mean_iou']['cloud_iou'] for x in results]
    mean_iou_snow = [x['mean_iou']['snow_iou'] for x in results]
    mean_iou_water = [x['mean_iou']['water_iou'] for x in results]
    mean_io_background = [x['mean_iou']['background_iou'] for x in results]

    plt.figure(figsize=(10, 5))
    plt.plot(labels, mean_iou_cloud, label="cloud")
    plt.plot(labels, mean_iou_snow, label="snow")
    plt.plot(labels, mean_iou_water, label="water")
    plt.plot(labels, mean_io_background, label="background")

    plt.xlabel("Date")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU per class")

    plt.legend(loc="lower left")

    plt.savefig(os.path.join(BASE_OUTPUT, f"{os.environ['TILE_NAME']}_mean_iou.png"))


def run_testing_on_date(s2_date, path_prefix):
    print(s2_date)

    result = {
        "mean_iou": None,
    }

    # Step 1: load predicted, ground truth and the mask used for training
    with rasterio.open(os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_mask_prediction.jp2")) as prediction_raw:
        print("prediction_raw.bounds", prediction_raw.bounds)

        # remove border of 256 pixels
        bounds = prediction_raw.bounds
        bounds = rasterio.coords.BoundingBox(
            left=bounds.left + 1024,
            bottom=bounds.bottom + 1024,
            right=bounds.right - 1024,
            top=bounds.top - 1024
        )
        window = prediction_raw.window(*bounds)
        prediction = prediction_raw.read(1, window=window)

        # load ground truth
        profile = get_profile(s2_date)

    s2_date_exoLabs_format = s2_date[0:4] + "-" + s2_date[4:6] + "-" + s2_date[6:8]
    exoLabs_folder = os.path.join(BASE_DIR, f"ExoLabs_classification_S2_{os.environ['TILE_NAME']}")

    exoLabs_file = \
        [f for f in os.listdir(exoLabs_folder) if
         f.startswith(f"S2_{os.environ['TILE_NAME']}_{s2_date_exoLabs_format}")][0]

    print(f"Loading exoLabs file: {exoLabs_file}")
    with rasterio.open(os.path.join(exoLabs_folder, exoLabs_file)) as exoLabs_raw:
        # read pixels form the same area as our model
        exo_labs_prediction_their_encoding = exoLabs_raw.read(1, window=window)
        print(f"ExoLabs prediction shape: {exo_labs_prediction_their_encoding.shape}")

        # convert pixel classes to same encoding as our model
        # 0 - background, no special class
        # 1 - snow
        # 2 - dense clouds
        # 3 - water
        # 4 - semi-transparent clouds

        # exolabs encoding:
        # notObserved = 0      (-)     - no data
        # noData = 1          (grey)   - unknown
        # darkFeatures = 2    (black)  - unknown
        # clouds = 3          (white)  - unknown
        # snow = 4            (red)    - snow
        # vegetation = 5      (green)  - no snow
        # water = 6           (blue)   - no snow
        # bareSoils = 7       (yellow) - no snow
        # glacierIce = 8      (cyan)   - no snow

        exo_labs_prediction = np.zeros(exo_labs_prediction_their_encoding.shape, dtype=np.uint8)
        exo_labs_prediction[exo_labs_prediction_their_encoding == 4] = 1
        exo_labs_prediction[exo_labs_prediction_their_encoding == 3] = 2
        exo_labs_prediction[exo_labs_prediction_their_encoding == 6] = 3
        exo_labs_prediction[exo_labs_prediction_their_encoding == 8] = 1

        # resample to same shape as our model
        exo_labs_prediction = cv2.resize(exo_labs_prediction, (prediction.shape[1], prediction.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

    with rasterio.open(os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_exolabs.jp2"), 'w',
                       **profile) as exolabs_save:
        exolabs_save.write(exo_labs_prediction, 1, window=window)

    with rasterio.open(os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_data_coverage.jp2")) as data_coverage_raw:
        data_coverage = data_coverage_raw.read(1, window=window)

    # add a safety margin of 256 pixels around every data_coverage == 0 pixel
    data_coverage = ~data_coverage
    data_coverage = cv2.dilate(data_coverage, np.ones((128, 128), np.uint8), iterations=1)
    data_coverage = ~data_coverage

    data_coverage[0:64, :] = 0
    data_coverage[-64:, :] = 0
    data_coverage[:, 0:64] = 0
    data_coverage[:, -64:] = 0

    prediction[data_coverage == 0] = 255

    # Step 3: calculate metrics
    print("\n\n\n=================\nTesting results:\n=================\n")

    print("Pixel counts (Our Prediction):")
    report_pixel_counts(prediction)

    print("Pixel counts (ExoLabs Prediction):")
    report_pixel_counts(exo_labs_prediction)

    __create_different_mask(window, profile, prediction, exo_labs_prediction, s2_date, path_prefix)

    # reduce to 1D arrays
    prediction = prediction.flatten()
    exo_labs_prediction = exo_labs_prediction.flatten()

    print("\n\n\n=================\nTesting results:\n=================\n")
    print("Report Similarity for Prediction vs. ExoLabs Prediction:")
    result['mean_iou'] = report_uio(exo_labs_prediction, prediction)
    report_dice(exo_labs_prediction, prediction)
    report_jaccard(exo_labs_prediction, prediction)
    print("\n\n\n=================\nTesting results:\n=================\n")

    return result


def report_jaccard(ground_truth, prediction):
    def __calculate_jaccard(ground_truth, prediction):
        intersection = np.sum(ground_truth * prediction)
        union = 1E-12 + np.sum(ground_truth) + np.sum(prediction) - intersection
        return intersection / union

    print("Class Jaccard:")

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
        union = 1E-12 + np.sum(ground_truth) + np.sum(prediction)
        return 2 * intersection / union

    print("Class Dice:")

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
        union = 1E-12 + np.logical_or(A, B)
        return np.sum(intersection) / np.sum(union)

    print("Class IOU:")
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

    return {
        'snow_iou': snow_iou,
        'cloud_iou': cloud_iou,
        'water_iou': water_iou,
        'background_iou': background_iou,
    }


def report_pixel_counts(mask):
    # mask may be a 2D array or a 1D array
    number_of_pixels = mask.size if mask.ndim == 1 else mask.shape[0] * mask.shape[1]

    print(f"- Total Number of pixels:\t{number_of_pixels:10d} ( 100.000 % )")

    numer_of_snow_pixels = np.sum(mask == 1)
    print(f"- Number of snow pixels:\t{numer_of_snow_pixels:10d} "
          f"( {(numer_of_snow_pixels / number_of_pixels * 100):02.3f} % )")

    numer_of_cloud_pixels = np.sum(mask == 2)
    print(f"- Number of cloud pixels:\t{numer_of_cloud_pixels:10d} "
          f"( {(numer_of_cloud_pixels / number_of_pixels * 100):02.3f} % )")

    if (numer_of_cloud_pixels / number_of_pixels * 100) > 85:
        print("WARNING: There are more than 85% cloud pixels in this image!")
        raise Exception("Too many cloud pixels!")

    numer_of_water_pixels = np.sum(mask == 3)
    print(f"- Number of water pixels:\t{numer_of_water_pixels:10d} "
          f"( {(numer_of_water_pixels / number_of_pixels * 100):02.3f} % )")

    numer_of_background_pixels = np.sum(mask == 0)
    print(f"- Number of background pixels:\t{numer_of_background_pixels:10d} "
          f"( {(numer_of_background_pixels / number_of_pixels * 100):02.3f} % )")

    print("")


def __create_different_mask(window, profile, my_pred, other_pred, s2_date, path_prefix, name="mask_diff"):
    # calculate the difference between the predicted mask and the training mask
    # and save it as a jp2 file

    assert my_pred.shape == other_pred.shape

    mask_with_invalid_values = np.logical_or(my_pred > 3, other_pred > 3)
    my_pred[mask_with_invalid_values] = 0
    other_pred[mask_with_invalid_values] = 0

    # calculate the difference (pixel wise)
    # define a lookup table for the output values
    # calculate the difference using vectorized operations and the lookup table
    lookup = np.array([0, 11, 12, 13, 1, 0, 14, 15, 2, 4, 0, 16, 3, 5, 6, 0])
    diff = lookup[my_pred * 4 + other_pred]
    diff = diff.astype('int8')

    # set invalid values to 255
    diff[mask_with_invalid_values] = 255

    # save the difference
    path = os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_{name}.jp2")

    profile.update({
        'nodata': 255,
    })

    with rasterio.open(path, 'w', **profile) as mask_file:
        mask_file.write(diff, 1, window=window)
