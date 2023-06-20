# This is a plain python version of the random_sampler Jupyter Notebook
# it is used to generate the training and testing datasets
import os

import numpy as np
from tqdm import tqdm

from config import SAMPLES_PER_DATE, DATASET_DIR, report_config, PATCH_SIZE, SELECTED_BANDS, AUXILIARY_DATA
from src.datahandler.NormalizedDataHandler import NormalizedDataHandler
from src.datahandler.patch_creator.RandomPatchCreator import RandomPatchCreator
from src.datahandler.satallite_reader.SentinelL1CReader import SentinelL1CReader
from src.python_helpers.pipeline_config import load_pipeline_config, get_dates


# ====================================
# ====================================
#  TODO: this is old code: Helper functions
# ====================================
# ====================================


# def plot_patch_centers(date, _patch_creator: SentinelDataLoader):
#     mask_data = _patch_creator.get_mask(date)
#     mask_coverage_data = _patch_creator.get_mask_coverage(date)
#
#     matplotlib.use('Agg')
#     plt.clf()
#     plt.imshow(mask_data, alpha=mask_coverage_data * 0.75)
#     plt.title(f'Overview of T{os.environ["TILE_NAME"]}')
#     plt.xticks([])
#     plt.yticks([])
#
#     for x, y in _patch_creator.get_patch_centers(date):
#         plt.gca().add_patch(patches.Rectangle((y, x), IMAGE_SIZE, IMAGE_SIZE, facecolor='r'))
#
#     # create folder for dataset centers
#     center_images_path = f"{RESULTS}/image_splitter"
#     if not os.path.exists(center_images_path):
#         os.makedirs(center_images_path)
#
#     print(f"Save center overview to {center_images_path}/{date}_centers.png\n")
#     plt.savefig(f"{center_images_path}/{date}_centers.png")


# ====================================
# ====================================
# END Helper functions
# ====================================
# ====================================

def main():
    pipeline_config = load_pipeline_config()

    # check if recreate_dataset is enabled
    if not pipeline_config['dataset']['recreate_dataset']:
        print("Dataset Creation is disabled. Exiting...")
        return

    assert pipeline_config['satellite'] == 'sentinel2', "Only Sentinel-2 is supported for dataset creation."

    # create target directories
    os.makedirs(f"{DATASET_DIR}/images", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/masks", exist_ok=True)

    # empty directories
    for file in os.listdir(f"{DATASET_DIR}/images"):
        os.remove(f"{DATASET_DIR}/images/{file}")
    for file in os.listdir(f"{DATASET_DIR}/masks"):
        os.remove(f"{DATASET_DIR}/masks/{file}")

    dates = get_dates(pipeline_config)
    tile_id = pipeline_config['tile_id']

    limit_dates = int(pipeline_config['dataset']['limit_dates'])
    if limit_dates > 0:
        dates = dates[:limit_dates]
        print(f"Limiting dates to {limit_dates} dates: {dates}")

    dataloader = NormalizedDataHandler(satellite_reader=SentinelL1CReader())

    patch_creator = RandomPatchCreator(
        dataloader=dataloader,
        patch_size=PATCH_SIZE,
        bands=SELECTED_BANDS,
        auxiliary_data=AUXILIARY_DATA
    )

    for date in tqdm(dates):
        print(f"Creating dataset from for scene {tile_id} - {date}...")

        # create patches
        for i in range(SAMPLES_PER_DATE):
            img_patch, mask_patch = patch_creator.get_random_patch(tile_id, date)
            np.save(f"{DATASET_DIR}/masks/{date}_{i}.npy", mask_patch)
            np.save(f"{DATASET_DIR}/images/{date}_{i}.npy", img_patch)

        # Show an overview of the centers
        patch_creator.plot_patch_centers(tile_id, date)

    pixel_count = patch_creator.get_class_distribution(tile_id)
    print(f"\n Total Class Distribution: {np.round(pixel_count, 5)} "
          f"« Background, Snow, Clouds, Water, Semi-Transparent Clouds\n")


if __name__ == '__main__':
    report_config()
    main()
