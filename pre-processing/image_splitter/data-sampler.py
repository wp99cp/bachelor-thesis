# This is a plain python version of the random_sampler Jupyter Notebook
# it is used to generate the training and testing datasets
import multiprocessing
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from joblib import Parallel, delayed
from matplotlib import patches
from tqdm import tqdm

from config import IMG_SIZE, SAMPLES_PER_DATE, NUM_ENCODED_CHANNELS, SELECTED_BANDS, BASE_DIR, TMP_DIR, MAKS_PATH, \
    DATASET_DIR, RESULTS

dates = os.listdir(BASE_DIR + MAKS_PATH)

# filter out ".gitignore"
dates = [d for d in dates if d != ".gitignore"]
dates.sort()
print(f"Found {len(dates)} dates")


# ====================================
# ====================================
# Helper functions
# ====================================
# ====================================

# Load the mask_coverage and the mask
def load_mask_coverage(_date):
    mask_coverage_path = BASE_DIR + MAKS_PATH + '/' + _date + '/mask_coverage.jp2'
    return rasterio.open(mask_coverage_path)


# Load the mask
def load_mask(_date):
    mask_path = BASE_DIR + MAKS_PATH + '/' + _date + '/mask.jp2'
    return rasterio.open(mask_path)


# sample a random patch from the mask_coverage
# where the mask_coverage is 1
def sample_patch(_mask_coverage, recursive=True):
    # sample a random patch
    _x = np.random.randint(0, _mask_coverage.shape[0] - IMG_SIZE)
    _y = np.random.randint(0, _mask_coverage.shape[1] - IMG_SIZE)

    # check if the patch is valid
    if np.sum(_mask_coverage[_x:_x + IMG_SIZE, _y:_y + IMG_SIZE]) == IMG_SIZE ** 2:
        return _x, _y
    else:
        if recursive:
            return sample_patch(_mask_coverage, recursive)
        else:
            return None


def colorize_mask(mask_):
    # Create color map with 4 colors
    cmap_ = np.array([
        [255, 255, 255],  # background
        [0, 0, 255],  # snow is blue
        [0, 255, 0],  # dense clouds are green
        [255, 0, 0],  # water is red
        [0, 160, 0]  # semi-dense clouds are dark green
    ])

    # convert to scalar type
    mask_ = np.clip(mask_, 0, NUM_ENCODED_CHANNELS)
    mask_ = mask_.astype(int)
    mask_ = cmap_[mask_]
    mask_ = mask_.astype(np.uint8)
    return cmap_, mask_


def open_date(_date):
    mask_coverage = load_mask_coverage(_date)
    mask = load_mask(_date)

    # Plot the mask_coverage and the mask
    mask_coverage_data = mask_coverage.read(1)
    mask_data = mask.read(1)

    B01 = f"T32TNS_{_date}_B01.jp2"
    B02 = f"T32TNS_{_date}_B02.jp2"
    B03 = f"T32TNS_{_date}_B03.jp2"
    B04 = f"T32TNS_{_date}_B04.jp2"
    B05 = f"T32TNS_{_date}_B05.jp2"
    B06 = f"T32TNS_{_date}_B06.jp2"
    B07 = f"T32TNS_{_date}_B07.jp2"
    B08 = f"T32TNS_{_date}_B08.jp2"
    B8A = f"T32TNS_{_date}_B8A.jp2"
    B09 = f"T32TNS_{_date}_B09.jp2"
    B10 = f"T32TNS_{_date}_B10.jp2"
    B11 = f"T32TNS_{_date}_B11.jp2"
    B12 = f"T32TNS_{_date}_B12.jp2"
    TCI = f"T32TNS_{_date}_TCI.jp2"

    # Search for a folder starting with "S2B_MSIL1C_$DATE"
    folders = [folder for folder in os.listdir(TMP_DIR) if f"_MSIL1C_{_date}" in folder]
    folder = folders[0]
    BASE_PATH = f"{TMP_DIR}/{folder}/GRANULE"
    sub_folders = os.listdir(BASE_PATH)
    BASE_PATH += '/' + sub_folders[0] + '/IMG_DATA'

    # print(f"Using {BASE_PATH} as base path")

    band_files = []

    # select bands based on SELECTED_BANDS
    for b in SELECTED_BANDS:
        # interpret b as a variable_name
        # and get the value of the variable
        band_files.append(eval(b))

    band_files = [f"{BASE_PATH}/{band}" for band in band_files]

    # open all the bands
    open_bands = [rasterio.open(band) for band in band_files]
    bands_data = [band.read(1) for band in open_bands]

    # upscale all bands to 10m resolution using cv2.resize
    b02_meta = rasterio.open(f"{BASE_PATH}/{B02}").meta
    image_with = b02_meta['width']
    image_height = b02_meta['height']
    bands_data = np.array(
        [cv2.resize(band, (image_with, image_height), interpolation=cv2.INTER_CUBIC) for band in bands_data]
    )

    print(f"  - Opened {len(bands_data)} bands for date {_date}.")
    # print(f"Shape of bands: {bands_data.shape}")

    # close all bands
    for band in open_bands:
        band.close()

    # Open all the bands
    return bands_data


def create_patches(_mask_coverage_data, _mask_data, _bands, _date_str):
    # get number of processes
    # check if environment NUM_PROCESSES is set
    # if not, use all available cores
    NUM_PROCESSES = os.environ.get('NUM_PROCESSES')
    num_processes = int(NUM_PROCESSES) if NUM_PROCESSES else multiprocessing.cpu_count()

    # print(f"Using {num_processes} processes.")

    def save_patch(i):
        coords = sample_patch(_mask_coverage_data)

        x, y = coords

        mask_patch = _mask_data[x:x + IMG_SIZE, y:y + IMG_SIZE]
        mask_patch = mask_patch * (255 / NUM_ENCODED_CHANNELS)

        # Count the number of pixels in each class
        _pixel_count = [0] * NUM_ENCODED_CHANNELS
        for j in range(NUM_ENCODED_CHANNELS):
            _pixel_count[j] += np.sum(mask_patch == j * (255 / NUM_ENCODED_CHANNELS))

        mask_patch_path = f"{DATASET_DIR}/masks/{_date_str}_{i}.png"
        img = Image.fromarray(mask_patch.astype(np.uint8))
        img.save(mask_patch_path)

        # load bands for img
        patch = _bands[:, x:x + IMG_SIZE, y:y + IMG_SIZE]

        # save the image as npy
        np.savez_compressed(f"{DATASET_DIR}/images/{_date_str}_{i}.npz", patch)

        return (x, y), _pixel_count

    print(f"    Start creating patches...")
    results = Parallel(n_jobs=num_processes)(delayed(save_patch)(i, ) for i in tqdm(range(SAMPLES_PER_DATE)))

    # unzip results
    _centers, pixel_counts = zip(*results)
    pixel_count = np.sum(pixel_counts, axis=0)

    normalized_pixel_count = np.array(pixel_count) / np.sum(pixel_count)
    print(f"    Class Distribution for {_date_str}: {np.round(normalized_pixel_count, 2)}")
    print("     « Background, Snow, Clouds, Water, Semi-Transparent Clouds")
    return _centers, normalized_pixel_count


# ====================================
# ====================================
# Main
# ====================================
# ====================================

def main():

    # create target directories
    os.makedirs(f"{DATASET_DIR}/images", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/masks", exist_ok=True)

    pixel_count = [0] * NUM_ENCODED_CHANNELS

    # Iterate over all dates
    for date in dates:

        date_str = date[0:4] + '-' + date[4:6] + '-' + date[6:8]
        print(f"\nStart processing Date: {date_str}")

        bands = open_date(date)
        bands = bands / 10_000 * 255
        # print(f"Bands loaded for {date_str}.")
        print(f"    Number of bands: {bands.shape}")

        print("    Loading mask and mask_coverage...")

        # Get the mask_coverage and the mask
        mask_coverage = load_mask_coverage(date)
        mask = load_mask(date)

        mask_data = mask.read(1)
        mask_coverage_data = mask_coverage.read(1)

        print(f"    Mask and mask_coverage loaded for {date_str}.")

        # print_sample(mask_coverage_data, mask_data, bands, date_str)
        centers, normalized_pixel_count = create_patches(mask_coverage_data, mask_data, bands, date_str)
        pixel_count += normalized_pixel_count

        print(f"    Finished processing Date: {date_str}")

        # Show an overview of the centers
        matplotlib.use('Agg')
        plt.clf()
        plt.imshow(mask_data, alpha=mask_coverage_data * 0.75)
        plt.title('Overview of T32TNS')
        plt.xticks([])
        plt.yticks([])
        for x, y in centers:
            plt.gca().add_patch(patches.Rectangle((y, x), IMG_SIZE, IMG_SIZE, facecolor='r'))

        # create folder for dataset centers
        center_images_path = f"{RESULTS}/image_splitter"
        if not os.path.exists(center_images_path):
            os.makedirs(center_images_path)

        plt.savefig(f"{center_images_path}/{date_str}_centers.png")

    # Show the distribution of the pixel count
    normalized_pixel_count = np.array(pixel_count) / np.sum(pixel_count)
    print(f"\n Total Class Distribution: {np.round(normalized_pixel_count, 5)}")
    print("« Background, Snow, Clouds, Water, Semi-Transparent Clouds")


if __name__ == '__main__':
    main()
