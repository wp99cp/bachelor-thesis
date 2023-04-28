# This is a plain python version of the random_sampler Jupyter Notebook
# it is used to generate the training and testing datasets
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patches
from tqdm import tqdm

from SentinelDataLoader import SentinelDataLoader
from config import IMAGE_SIZE, SAMPLES_PER_DATE, MAKS_PATH, \
    DATASET_DIR, RESULTS, report_config, EXTRACTED_RAW_DATA, SELECTED_BANDS

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/helper-scripts/python_helpers')
# noinspection PyUnresolvedReferences
from pipeline_config import load_pipeline_config, get_dates

report_config()

pipeline_config = load_pipeline_config()
dates = get_dates(pipeline_config, pipeline_step='dataset')


# ====================================
# ====================================
# Helper functions
# ====================================
# ====================================

def save_patch(i, _patch_creator, _date):
    # create a random patch
    img_patch, mask_patch = _patch_creator.random_patch(_date)

    mask_patch_path = f"{DATASET_DIR}/masks/{_date}_{i}.png"
    img = Image.fromarray(mask_patch)
    img.save(mask_patch_path)

    # save the image as npy
    np.save(f"{DATASET_DIR}/images/{_date}_{i}.npy", img_patch)


def create_patches(_patch_creator, _date):
    print(f"    Start creating patches...")

    def __create_patches(i):
        save_patch(i, _patch_creator, _date)

    num_workers = 8
    with ThreadPoolExecutor(num_workers) as executor:
        list(tqdm(executor.map(__create_patches, range(SAMPLES_PER_DATE)), total=SAMPLES_PER_DATE))

    pixel_count = _patch_creator.get_PixelClassCounter().get_class_distribution(_date)
    print(f"    Class Distribution for {_date}: {np.round(pixel_count, 2)}")
    print("     « Background, Snow, Clouds, Water, Semi-Transparent Clouds")


# ====================================
# ====================================
# Main
# ====================================
# ====================================

def main():
    # create target directories
    os.makedirs(f"{DATASET_DIR}/images", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/masks", exist_ok=True)

    patch_creator = SentinelDataLoader(dates=dates,
                                       mask_base_dir=MAKS_PATH,
                                       raw_data_base_dir=EXTRACTED_RAW_DATA,
                                       selected_bands=SELECTED_BANDS)

    # Iterate over all dates
    for i, date in enumerate(dates):
        print(f"\nStart processing Date: {date}. That's {i + 1} of {len(dates)}.")

        # load the date into memory
        # create and save the patches
        patch_creator.open_date(date)
        create_patches(patch_creator, date)

        # Show an overview of the centers
        plot_patch_centers(date, patch_creator)

        # close the date to free memory
        patch_creator.close_date(date)

        print(f"    Finished processing Date: {date}")

    pixel_count = patch_creator.get_PixelClassCounter().get_class_distribution()
    print(f"\n Total Class Distribution: {np.round(pixel_count, 5)} "
          f"« Background, Snow, Clouds, Water, Semi-Transparent Clouds\n")


def plot_patch_centers(date, _patch_creator: SentinelDataLoader):
    mask_data = _patch_creator.get_mask(date)
    mask_coverage_data = _patch_creator.get_mask_coverage(date)

    matplotlib.use('Agg')
    plt.clf()
    plt.imshow(mask_data, alpha=mask_coverage_data * 0.75)
    plt.title('Overview of T32TNS')
    plt.xticks([])
    plt.yticks([])

    for x, y in _patch_creator.get_patch_centers(date):
        plt.gca().add_patch(patches.Rectangle((y, x), IMAGE_SIZE, IMAGE_SIZE, facecolor='r'))

    # create folder for dataset centers
    center_images_path = f"{RESULTS}/image_splitter"
    if not os.path.exists(center_images_path):
        os.makedirs(center_images_path)

    print(f"Save center overview to {center_images_path}/{date}_centers.png\n")
    plt.savefig(f"{center_images_path}/{date}_centers.png")


if __name__ == '__main__':
    main()
