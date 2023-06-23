# import the necessary packages form the pre-processing/dataset_creator
import os
from datetime import datetime

import cv2
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.enums import Resampling

from src.datahandler.WindowGenerator import WindowGenerator
from src.models.unet.configs.config import BASE_OUTPUT
from src.python_helpers.pipeline_config import get_dates


def run_testing(pipeline_config, model_file_name='unet'):
    print("[INFO] running testing...")

    dates = get_dates(pipeline_config)
    print(f"[INFO] Run inference on dates: {dates}")

    tile_name = pipeline_config["tile_id"]
    print(f"[INFO] Tile name: {tile_name}")

    results = []

    skipped_dates = []

    if pipeline_config["testing"]["use_cache"] == 0:
        for s2_date in dates:

            try:
                model_name = model_file_name.split(".")[0]
                path_prefix = os.path.join(f"{tile_name}_{s2_date}_pred", model_name)

                result = run_testing_on_date(s2_date, tile_name, path_prefix, pipeline_config)
                result['date'] = s2_date[0:4] + "-" + s2_date[4:6] + "-" + s2_date[6:8]
                results.append(result)

            except Exception as e:
                print(e)
                print(f"[INFO] Skipping date {s2_date}")

                skipped_dates.append(s2_date)

                continue

        print(f"\n\n\n====================\n====================\n====================\n\n\n")
        print(results)

        print(f"\n\n\n====================\n====================\n====================\n\n\n")
        print(f"Skipped dates for tile_id={tile_name}:")
        print(skipped_dates)

    # sort results by date
    results = sorted(results, key=lambda k: k['date'])

    # save results to file
    if pipeline_config["testing"]["use_cache"] == 0:
        np.save(os.path.join(BASE_OUTPUT, f"difference_to_exolabs_{tile_name}.npy"), results)

    # load results from file
    if pipeline_config["testing"]["use_cache"] == 1:
        results = np.load(os.path.join(BASE_OUTPUT, f"difference_to_exolabs_{tile_name}.npy"),
                          allow_pickle=True)

    labels = [x['date'] for x in results]
    labels = [datetime.strptime(d, '%Y-%m-%d') for d in labels]

    multiclass_iou = [x['iou']['multiclass_iou'] for x in results]

    mean_iou_cloud = [x['iou']['cloud_iou'] for x in results]
    mean_iou_snow = [x['iou']['snow_iou'] for x in results]
    mean_iou_water = [x['iou']['water_iou'] for x in results]
    mean_io_background = [x['iou']['background_iou'] for x in results]

    plt.figure(figsize=(10, 5))
    plt.plot(labels, mean_iou_cloud, label="cloud")
    plt.plot(labels, mean_iou_snow, label="snow")
    plt.plot(labels, mean_iou_water, label="water")
    plt.plot(labels, mean_io_background, label="background")

    plt.xlabel("Date")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU per class")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(BASE_OUTPUT, f"{tile_name}_mean_iou.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(labels, multiclass_iou, label="IoU")
    plt.xlabel("Date")
    plt.ylabel("IoU")
    plt.title("Multiclass IoU")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(BASE_OUTPUT, f"{tile_name}_multiclass_iou.png"))


def run_testing_on_date(s2_date, tile_name, path_prefix, pipeline_config):
    print(s2_date)

    result = {
        "iou": None,
        "dice": None,
        "jaccard": None
    }

    # Step 1: load predicted, ground truth and the mask used for training
    with rasterio.open(os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_mask_prediction.jp2")) as prediction_raw:
        window_generator = WindowGenerator(prediction_raw.transform)
        window = window_generator.get_window(tile_id=tile_name)

        prediction = prediction_raw.read(
            1,
            out_shape=(
                prediction_raw.count,
                int(window.height),
                int(window.width)
            ),
            window=window,
            resampling=Resampling.nearest
        )

        profile = {
            'driver': 'GTiff',
            'dtype': np.uint8,
            'nodata': 0,
            'width': prediction_raw.width,
            'height': prediction_raw.height,
            'count': 1,
            'crs': prediction_raw.crs,
            'transform': prediction_raw.transform,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'lzw',
        }

    s2_date_exoLabs_format = s2_date[0:4] + "-" + s2_date[4:6] + "-" + s2_date[6:8]

    exoLabs_folder = os.path.join(
        os.environ['DATA_DIR'],
        pipeline_config['satellite'],
        "exoLabs",
        f"T{pipeline_config['tile_id']}")

    exoLabs_file = \
        [f for f in os.listdir(exoLabs_folder) if
         f.startswith(f"S2_{tile_name}_{s2_date_exoLabs_format}")][0]

    print(f"Loading exoLabs file: {exoLabs_file}")
    with rasterio.open(os.path.join(exoLabs_folder, exoLabs_file)) as exoLabs_raw:
        window_generator = WindowGenerator(exoLabs_raw.transform)
        window = window_generator.get_window(tile_id=tile_name)

        exo_labs_prediction_their_encoding = exoLabs_raw.read(
            1,
            out_shape=(
                1,
                int(window.height),
                int(window.width)
            ),
            window=window,
            resampling=Resampling.nearest
        )

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

    with rasterio.open(os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_data_coverage.jp2")) as data_coverage_raw:
        window_generator = WindowGenerator(data_coverage_raw.transform)
        window = window_generator.get_window(tile_id=tile_name)
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
    result['iou'] = report_uio(exo_labs_prediction, prediction)
    result['dice'] = report_dice(exo_labs_prediction, prediction)
    result['jaccard'] = report_jaccard(exo_labs_prediction, prediction)
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

    return {
        'snow_jaccard': snow_jaccard,
        'cloud_jaccard': cloud_jaccard,
        'water_jaccard': water_jaccard,
        'background_jaccard': background_jaccard,
        'mean_jaccard': (snow_jaccard + cloud_jaccard + water_jaccard) / 3
    }


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

    return {
        'snow_dice': snow_dice,
        'cloud_dice': cloud_dice,
        'water_dice': water_dice,
        'background_dice': background_dice,
        'mean_dice': (snow_dice + cloud_dice + water_dice) / 3
    }


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

    multiclass_intersection = np.logical_and(ground_truth == 0, prediction == 0) + \
                              np.logical_and(ground_truth == 1, prediction == 1) + \
                              np.logical_and(ground_truth == 2, prediction == 2) + \
                              np.logical_and(ground_truth == 3, prediction == 3)
    multiclass_union = 1E-12 + np.logical_or(ground_truth == 0, prediction == 0) + \
                       np.logical_or(ground_truth == 1, prediction == 1) + \
                       np.logical_or(ground_truth == 2, prediction == 2) + \
                       np.logical_or(ground_truth == 3, prediction == 3)
    multiclass_iou = np.sum(multiclass_intersection) / np.sum(multiclass_union)
    print(f"Multiclass IOU (incl. background):\t{multiclass_iou:02.3f}")
    print("")

    return {
        'snow_iou': snow_iou,
        'cloud_iou': cloud_iou,
        'water_iou': water_iou,
        'background_iou': background_iou,
        'mean_iou': (snow_iou + cloud_iou + water_iou) / 3,
        'multiclass_iou': multiclass_iou
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
        # raise Exception("Too many cloud pixels!")

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

    ##########################
    # create simplified mask
    ##########################
    print("Creating simplified mask...")
    print("Shape of diff: ", diff.shape)
    diff_rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    STRETCH_FACTOR = 18
    diff_rgb[:, :, 0] = diff * STRETCH_FACTOR
    diff_rgb[:, :, 1] = diff * STRETCH_FACTOR
    diff_rgb[:, :, 2] = diff * STRETCH_FACTOR

    blurred_diff = cv2.medianBlur(diff_rgb, 5)

    kernel = np.ones((16, 16), np.uint8)
    erosion = cv2.erode(blurred_diff, kernel, iterations=1)
    output = cv2.dilate(erosion, kernel, iterations=1)
    print("Shape of output: ", output.shape)

    output_gray = output[:, :, 0] / STRETCH_FACTOR
    output_gray = output_gray.astype('uint8')

    path = os.path.join(BASE_OUTPUT, path_prefix, f"{s2_date}_{name}_simplified.jp2")
    with rasterio.open(path, 'w', **profile) as mask_file:
        mask_file.write(output_gray, 1, window=window)
