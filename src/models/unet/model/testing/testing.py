# import the necessary packages form the pre-processing/dataset_creator
import os
from datetime import datetime

import cv2
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.enums import Resampling

from src.datahandler.WindowGenerator import WindowGenerator
from src.models.unet.configs.config import BASE_OUTPUT, AUXILIARY_DATA_DIR
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

    print(f"\n\n\n====================\n====================\n====================\n\n\n")
    print(f"\n\n\n====================\n====================\n====================\n\n\n")
    print(f"\n\n\n====================\n====================\n====================\n\n\n")
    print(f"\n\n\n====================\n====================\n====================\n\n\n")

    # sort results by date
    results = sorted(results, key=lambda k: k['date'])

    # if limited to class
    limited = ''
    if pipeline_config["limit_to_landcover"] == 1:
        limited = '_limited'
        for i in pipeline_config["landcover_classes"]:
            limited += f"_{str(i)}"

    # save results to file
    if pipeline_config["testing"]["use_cache"] == 0:
        np.save(os.path.join(BASE_OUTPUT, f"difference_to_exolabs_{tile_name}{limited}.npy"), results)

    # load results from file
    if pipeline_config["testing"]["use_cache"] == 1:
        results = np.load(os.path.join(BASE_OUTPUT, f"difference_to_exolabs_{tile_name}{limited}.npy"),
                          allow_pickle=True)

    # remove corrupted dates: 2022-02-20
    results = [x for x in results if x['date'] != '2022-02-20']

    # results = [x for x in results if x['date'] in ['2021-01-11', '2021-01-16', '2021-01-21', '2021-01-26', '2021-01-31', '2021-02-15', '2021-02-20'
    #                                                      '2021-02-25', '2021-03-02', '2021-03-07', '2021-03-22', '2021-04-01', '2021-04-06', '2021-04-16',
    #                                                      '2021-05-26', '2021,05,31', '2021-06-10', '2021-06-15', '2021-06-25', '2021-07-05', '2021-07-10',
    #                                                      '2021-07-20', '2021-07-30', '2021-08-14', '2021-08-19', '2021-08-24', '2021-09-03', '2021-09-08',
    #                                                      '2021-09-13', '2021-09-18', '2021-10-08', '2021-10-13', '2021-10-18', '2021-10-28', '2021-11-02',
    #                                                      '2021-11-12'] ]

    # results = [x for x in results if '2021-' not in x['date']]



    # print dates with multiclass_iou below
    for result in results:
        threshold = 0.5
        if result['iou']['multiclass_iou'] < threshold:
            print(f"Date {result['date']} has multiclass_iou below {threshold}")
            print(f"Multiclass IoU: {result['iou']['multiclass_iou']}")

    print(f"Count n={len(results)}")

    labels = [x['date'] for x in results]
    labels = [datetime.strptime(d, '%Y-%m-%d') for d in labels]

    multiclass_iou = [x['iou']['multiclass_iou'] for x in results]
    mean_iou = [
        (x['iou']['snow_iou'] + x['iou']['cloud_iou'] + x['iou']['background_iou'] + x['iou']['water_iou']) / 4 for x in
        results]
    print(f"Mean multiclass IoU: {np.mean(multiclass_iou)}")
    print(f"Mean mean IoU: {np.mean(mean_iou)}")


    # print dates with multiclass_iou below
    # for result, miou in zip(results, mean_iou):
    #     threshold = 0.6
    #     threshold_top = 0.6
    #
    #     total_pixel_exolabs = result['report_pixel_counts_exo_labs_prediction']['number_of_pixels_with_data']
    #     cloud_pixel_exolabs = result['report_pixel_counts_exo_labs_prediction']['number_of_cloud_pixels']
    #
    #     if miou < threshold and result['iou']['multiclass_iou'] > threshold_top and cloud_pixel_exolabs / total_pixel_exolabs < 0.6:
    #         print(f"Date {result['date']} has mean iou below {threshold}")
    #         print(f"mean IoU: {miou} vs Mean multiclass IoU: {result['iou']['multiclass_iou']}")

    mean_iou_cloud = [x['iou']['cloud_iou'] for x in results]
    mean_iou_snow = [x['iou']['snow_iou'] for x in results]
    mean_iou_water = [x['iou']['water_iou'] for x in results]
    mean_io_background = [x['iou']['background_iou'] for x in results]

    print(f"Mean IoU cloud: {np.mean(mean_iou_cloud)}")
    print(f"Mean IoU snow: {np.mean(mean_iou_snow)}")
    print(f"Mean IoU water: {np.mean(mean_iou_water)}")
    print(f"Mean IoU background: {np.mean(mean_io_background)}")

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

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    plt.figure(figsize=(10, 5))
    # draw a rectangle to indicate the training period

    cloudy_points = []
    partially_cloudy_points = []
    wrong_cloudy_points = []

    normal_points = []
    THRESHOLD = 0.85

    mean_cloud_coverage = 0

    for scene in results:

        total_pixel = scene['report_pixel_counts_prediction']['number_of_pixels_with_data']
        cloud_pixel = scene['report_pixel_counts_prediction']['number_of_cloud_pixels']

        mean_cloud_coverage += cloud_pixel / total_pixel

        total_pixel_exolabs = scene['report_pixel_counts_exo_labs_prediction']['number_of_pixels_with_data']
        cloud_pixel_exolabs = scene['report_pixel_counts_exo_labs_prediction']['number_of_cloud_pixels']

        if cloud_pixel / total_pixel > THRESHOLD and cloud_pixel_exolabs / total_pixel_exolabs > THRESHOLD:

            cloudy_points.append((datetime.strptime(scene['date'], '%Y-%m-%d'),
                                  scene['iou']['multiclass_iou']))

        elif cloud_pixel / total_pixel > THRESHOLD:

            partially_cloudy_points.append((datetime.strptime(scene['date'], '%Y-%m-%d'),
                                            scene['iou']['multiclass_iou']))

        elif cloud_pixel_exolabs / total_pixel_exolabs > THRESHOLD:
            wrong_cloudy_points.append((datetime.strptime(scene['date'], '%Y-%m-%d'),
                                        scene['iou']['multiclass_iou']))

        else:
            normal_points.append((datetime.strptime(scene['date'], '%Y-%m-%d'),
                                  scene['iou']['multiclass_iou']))

    print(f"Mean cloud coverage: {mean_cloud_coverage / len(results)}")

    # combine with partially_cloudy_points
    manual_annotated_points = [('2021-01-25', 0.43), ('2021-03-28', 0.24)]

    # 32TNS: [('2021-01-06', 0.84), ('2021-01-11', 0.85), ('2021-03-12', 0.84), ('2022-11-02', 0.86), ('2022-12-27', 0.83)]
    # 13TDE: [('2021-01-11', 0.741), ('2021-02-20', 0.795), ('2021-08-09', 0.874)]
    # 07VEH: [('2021-01-25', 0.43), ('2021-03-28', 0.24)]
    # 32VMP: []

    manual_annotated_points.extend(partially_cloudy_points)
    print(partially_cloudy_points)



    plt.plot(labels, multiclass_iou, label="multi-class IoU", color='slategray', zorder=1)
    # plt.plot(labels, mean_iou, label="mean IoU", zorder=1)

    plt.scatter([x[0] for x in cloudy_points], [x[1] for x in cloudy_points], color='green',
                marker="X", label=f"cloudy scene (>{THRESHOLD}%) (both models agree)", zorder=2)
    plt.scatter([x[0] for x in partially_cloudy_points], [x[1] for x in partially_cloudy_points], color='orange',
                marker="P", label=f"cloudy scene (>{THRESHOLD}%) (our model only)", zorder=2)
    plt.scatter([x[0] for x in wrong_cloudy_points], [x[1] for x in wrong_cloudy_points], color='blue',
                marker="*", label=f"cloudy scene (>{THRESHOLD}%) (ExoLabs only)", zorder=2)
    plt.scatter([x[0] for x in manual_annotated_points], [x[1] for x in manual_annotated_points], color='red',
                s=100, facecolors='none', edgecolors='r', zorder=2, label="scenes mentioned")

    # training_dates = [x for x in results if x['date'] in [ '2021-01-11', '2021-01-16', '2021-01-21', '2021-01-26', '2021-01-31', '2021-02-15', '2021-02-20'
    #                                                      '2021-02-25', '2021-03-02', '2021-03-07', '2021-03-22', '2021-04-01', '2021-04-06', '2021-04-16',
    #                                                      '2021-05-26', '2021,05,31', '2021-06-10', '2021-06-15', '2021-06-25', '2021-07-05', '2021-07-10',
    #                                                      '2021-07-20', '2021-07-30', '2021-08-14', '2021-08-19', '2021-08-24', '2021-09-03', '2021-09-08',
    #                                                      '2021-09-13', '2021-09-18', '2021-10-08', '2021-10-13', '2021-10-18', '2021-10-28', '2021-11-02',
    #                                                      '2021-11-12'] ]
    #
    # plt.scatter([x['date'] for x in training_dates], [x['iou']['multiclass_iou'] for x in training_dates],
    #             s=20, facecolors='none', edgecolors='violet', zorder=2, label="training scenes")

    mean = np.mean([x[1] for x in normal_points])
    std = np.std([x[1] for x in normal_points])
    plt.axhspan(mean - std, mean + std, color='green', alpha=0.1, label="mean mcIoU +/- std (non-cloudy scenes)")
    # plt.axhspan(0, 0.8, color='orange', alpha=0.1, label="significant difference")

    print("normal points avg: ", mean)
    print("normal points std: ", std)

    plt.xlabel("Date")
    plt.ylabel("mcIoU")
    plt.ylim(0.2, 1.01)
    plt.title(f"Multi-Class IoU for {tile_name}")
    plt.legend(loc="lower right")
    plt.legend(ncol=1)
    plt.savefig(os.path.join(BASE_OUTPUT, f"{tile_name}_multiclass_iou.png"))


def run_testing_on_date(s2_date, tile_name, path_prefix, pipeline_config):
    print(s2_date)

    result = {
        "iou": None,
        "dice": None,
        "jaccard": None
    }

    # Step 1: load predicted, ground truth and the mask used for training
    print("Opening file with name: " + os.path.join(BASE_OUTPUT, path_prefix,
                                                    f"{s2_date}_{pipeline_config['testing']['prediction_name']}.jp2"))
    with rasterio.open(os.path.join(BASE_OUTPUT, path_prefix,
                                    f"{s2_date}_{pipeline_config['testing']['prediction_name']}.jp2")) as prediction_raw:
        window_generator = WindowGenerator(prediction_raw.transform)
        window = window_generator.get_window(tile_id=tile_name)

        prediction = prediction_raw.read(
            1,
            out_shape=(
                prediction_raw.count,
                int(window.width),
                int(window.height)
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

    exo_labs_prediction = np.zeros(prediction.shape, dtype=np.uint8)

    if pipeline_config["testing"]["compare_to_prediction"] == 0:
        print(f"Loading exoLabs file: {exoLabs_file}")
        with rasterio.open(os.path.join(exoLabs_folder, exoLabs_file)) as exoLabs_raw:
            window_generator = WindowGenerator(exoLabs_raw.transform)
            window = window_generator.get_window(tile_id=tile_name)

            exo_labs_prediction_their_encoding = exoLabs_raw.read(
                1,
                out_shape=(
                    1,
                    int(window.width),
                    int(window.height)
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

            exo_labs_prediction[exo_labs_prediction_their_encoding == 4] = 1
            exo_labs_prediction[exo_labs_prediction_their_encoding == 3] = 2
            exo_labs_prediction[exo_labs_prediction_their_encoding == 6] = 3
            exo_labs_prediction[exo_labs_prediction_their_encoding == 8] = 1

    else:
        path = os.path.join(BASE_OUTPUT, path_prefix,
                            f"{s2_date}_{pipeline_config['testing']['prediction_name_other']}.jp2")
        print(f"Loading other prediction file: {path}")
        with rasterio.open(os.path.join(BASE_OUTPUT, path_prefix,
                                        f"{s2_date}_{pipeline_config['testing']['prediction_name_other']}.jp2")) as prediction_raw:
            window_generator = WindowGenerator(prediction_raw.transform)
            window = window_generator.get_window(tile_id=tile_name)

            exo_labs_prediction = prediction_raw.read(
                1,
                out_shape=(
                    prediction_raw.count,
                    int(window.width),
                    int(window.height)
                ),
                window=window,
                resampling=Resampling.nearest
            )

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

    if pipeline_config["limit_to_landcover"] == 1:
        landcover_file = pipeline_config["landcover_file"]
        landcover_file = os.path.join(AUXILIARY_DATA_DIR, f"T{tile_name}", landcover_file)
        print(f"Loading landcover file: {landcover_file}")

        with rasterio.open(landcover_file) as landcover_raw:
            window_generator = WindowGenerator(landcover_raw.transform)
            window_landcover = window_generator.get_window(tile_id=tile_name)
            landcover = landcover_raw.read(1, window=window_landcover)
            print(f"Landcover shape: {landcover.shape}")

            data_coverage_landcover_mask = np.zeros(data_coverage.shape, dtype=np.uint8)
            for i in pipeline_config["landcover_classes"]:
                print(f"Removing landcover class {i} from data coverage")
                data_coverage_landcover_mask[landcover == i] = 1

            data_coverage &= data_coverage_landcover_mask

    prediction[data_coverage == 0] = 255
    exo_labs_prediction[data_coverage == 0] = 255

    # Step 3: calculate metrics
    print("\n\n\n=================\nTesting results:\n=================\n")

    print("Pixel counts (Our Prediction):")
    result['report_pixel_counts_prediction'] = report_pixel_counts(prediction, data_coverage)

    print("Pixel counts (ExoLabs Prediction):")
    result['report_pixel_counts_exo_labs_prediction'] = report_pixel_counts(exo_labs_prediction, data_coverage)

    mask_diff_name = "mask_diff" + ("_" +
                                    pipeline_config['testing']['difference_map_postfix'] if "difference_map_postfix" in
                                                                                            pipeline_config[
                                                                                                "testing"] else "")
    __create_different_mask(window, profile, prediction, exo_labs_prediction, s2_date, path_prefix, name=mask_diff_name)

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
    multiclass_union = np.logical_or(ground_truth == 0, prediction == 0) + \
                       np.logical_or(ground_truth == 1, prediction == 1) + \
                       np.logical_or(ground_truth == 2, prediction == 2) + \
                       np.logical_or(ground_truth == 3, prediction == 3)
    multiclass_iou = np.sum(multiclass_intersection) / (np.sum(multiclass_union) + 1E-12)
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


def report_pixel_counts(mask, data_coverage):
    # mask may be a 2D array or a 1D array
    number_of_pixels = mask.size if mask.ndim == 1 else mask.shape[0] * mask.shape[1]

    # number of pixesl with data_coverage == 1
    number_of_pixels_with_data = np.sum(data_coverage == 1)

    print(f"- Total Number of pixels:\t\t{number_of_pixels:10d} ( 100.000 % )")
    print(f"- Number of Pixels with valid data:\t{number_of_pixels_with_data:10d} "
          f"( {(number_of_pixels_with_data / number_of_pixels * 100):02.3f} % )")

    numer_of_snow_pixels = np.sum(mask == 1)
    print(f"- Number of snow pixels:\t\t{numer_of_snow_pixels:10d} "
          f"( {(numer_of_snow_pixels / number_of_pixels * 100):02.3f} % )")

    numer_of_cloud_pixels = np.sum(mask == 2)
    print(f"- Number of cloud pixels:\t\t{numer_of_cloud_pixels:10d} "
          f"( {(numer_of_cloud_pixels / number_of_pixels * 100):02.3f} % )")

    if (numer_of_cloud_pixels / number_of_pixels * 100) > 85:
        print(" » WARNING: There are more than 85% cloud pixels in this image!")
        # raise Exception("Too many cloud pixels!")

    numer_of_water_pixels = np.sum(mask == 3)
    print(f"- Number of water pixels:\t\t{numer_of_water_pixels:10d} "
          f"( {(numer_of_water_pixels / number_of_pixels * 100):02.3f} % )")

    numer_of_background_pixels = np.sum(mask == 0)
    print(f"- Number of background pixels:\t\t{numer_of_background_pixels:10d} "
          f"( {(numer_of_background_pixels / number_of_pixels * 100):02.3f} % )")

    print("")

    return {
        'number_of_pixels': number_of_pixels,
        'number_of_pixels_with_data': number_of_pixels_with_data,
        'number_of_snow_pixels': numer_of_snow_pixels,
        'number_of_cloud_pixels': numer_of_cloud_pixels,
        'number_of_water_pixels': numer_of_water_pixels,
    }


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
