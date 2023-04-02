import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import tqdm
from imutils import paths
from numpy.random import multivariate_normal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as tsfm

from DataLoader.SegmentationDiskDataset import SegmentationDiskDataset
from DataLoader.SegmentationLiveDataset import SegmentationLiveDataset
from augmentation.Augmentation import Augmentation
from augmentation.ChannelDropout import ChannelDropout
from augmentation.HorizontalFlip import HorizontalFlip
from augmentation.RandomErasing import RandomErasing
from augmentation.VerticalFlip import VerticalFlip
from configs.config import report_config, IMAGE_DATASET_PATH, MASK_DATASET_PATH, TEST_SPLIT, BATCH_SIZE, PIN_MEMORY, \
    DEVICE, BASE_OUTPUT, ENABLE_DATA_AUGMENTATION, IMAGE_FLIP_PROB, CHANNEL_DROPOUT_PROB, \
    COVERED_PATCH_SIZE_MIN, COVERED_PATCH_SIZE_MAX, LIMIT_DATASET_SIZE, BATCH_PREFETCHING, NUM_DATA_LOADER_WORKERS, \
    THRESHOLD, NUM_CLASSES, IMAGE_SIZE
from model.Model import UNet
from model.inference import make_predictions
from training import train_unet

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/helper-scripts/python_helpers')
# noinspection PyUnresolvedReferences
from pipeline_config import load_pipeline_config, get_dates

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/pre-processing/image_splitter')
# noinspection PyUnresolvedReferences
from RandomPatchCreator import RandomPatchCreator


def load_data():
    # load the image and mask filepaths in a sorted manner
    image_paths = sorted(list(paths.list_files(IMAGE_DATASET_PATH, validExts=("npy",))))
    mask_paths = sorted(list(paths.list_images(MASK_DATASET_PATH)))

    if LIMIT_DATASET_SIZE > 0:
        image_paths = image_paths[:LIMIT_DATASET_SIZE]
        mask_paths = mask_paths[:LIMIT_DATASET_SIZE]

    print(f"[INFO] found a total of {len(image_paths)} images in '{IMAGE_DATASET_PATH}'.")
    print(f"[INFO] found a total of {len(mask_paths)} masks in '{MASK_DATASET_PATH}'.")
    assert len(image_paths) == len(mask_paths), "Number of images and masks must match."

    # define transformations
    transforms = tsfm.Compose([tsfm.ToTensor()])
    augmentations: list[Augmentation] = [
        HorizontalFlip(prob=IMAGE_FLIP_PROB),
        VerticalFlip(prob=IMAGE_FLIP_PROB),
        ChannelDropout(prob=CHANNEL_DROPOUT_PROB),
        RandomErasing(prob=CHANNEL_DROPOUT_PROB, min_size=COVERED_PATCH_SIZE_MIN, max_size=COVERED_PATCH_SIZE_MAX)
    ]

    trainImages = []
    testImages = []

    # create the train and test datasets
    # the live dataset creation can be enabled using the create_on_the_fly
    # setting in th pipeline_config.json
    if int(os.environ.get("LIVE_DATASET", 0)) == 1:

        print(f"[INFO] creating the dataset on the fly. (LIVE_DATASET=1)")

        pipeline_config = load_pipeline_config()
        dates = get_dates(pipeline_config, pipeline_step='dataset')

        train_ds = SegmentationLiveDataset(dates=dates, transforms=transforms,
                                           apply_augmentations=ENABLE_DATA_AUGMENTATION,
                                           augmentations=augmentations)

        train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, prefetch_factor=BATCH_PREFETCHING,
                                  pin_memory=PIN_MEMORY, num_workers=NUM_DATA_LOADER_WORKERS)

        test_ds = train_ds
        test_loader = train_loader

    else:

        # partition the data into training and testing splits using 85% of
        # the data for training and the remaining 15% for testing
        split = train_test_split(image_paths, mask_paths, test_size=TEST_SPLIT, random_state=42)

        # unpack the data split
        (trainImages, testImages) = split[:2]
        (trainMasks, testMasks) = split[2:]

        print(f"[INFO] loading the dataset from disk. (LIVE_DATASET=0)")

        train_ds = SegmentationDiskDataset(image_paths=trainImages, mask_paths=trainMasks, transforms=transforms,
                                           apply_augmentations=ENABLE_DATA_AUGMENTATION, augmentations=augmentations)
        test_ds = SegmentationDiskDataset(image_paths=testImages, mask_paths=testMasks,
                                          transforms=transforms, apply_augmentations=False)

        # create the training and test data loaders
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, prefetch_factor=BATCH_PREFETCHING,
                                  pin_memory=PIN_MEMORY, num_workers=NUM_DATA_LOADER_WORKERS)
        test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, prefetch_factor=BATCH_PREFETCHING,
                                 pin_memory=PIN_MEMORY, num_workers=NUM_DATA_LOADER_WORKERS)

    print(f"[INFO] loaded {len(train_ds)} examples in the train set.")
    print(f"[INFO] loaded {len(test_ds)} examples in the test set.")
    print(f"\n")

    return train_loader, test_loader, train_ds, test_ds, trainImages, testImages


def print_data_sample(train_loader):
    # show a sample image and mask
    sample = next(iter(train_loader))

    img = sample[0][0].permute(1, 2, 0)
    masks = sample[1][0].permute(1, 2, 0)

    print(f"Image shape: {img.shape}")

    # use matplotlib to display the image and mask
    # show a legend for the mask next to the image
    # i.g. blue = snow, green = clouds, red = water
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img[:, :, 0:3])
    ax[0].set_title("Image")

    ax[1].imshow(masks[:, :, 0])
    ax[1].set_title("Mask")

    sample_data_path = os.path.join(BASE_OUTPUT, "sample_data.png")
    plt.savefig(sample_data_path)


def main():
    report_config()

    # check if the flag "--retrain" is set
    # if so, train the model
    emergency_stop = False

    if "--retrain" in sys.argv:
        train_loader, test_loader, train_ds, test_ds, _, test_images = load_data()
        print_data_sample(train_loader)

        print("[INFO] retraining model...")
        emergency_stop = train_unet(train_loader, test_loader, train_ds, test_ds)

        # load the image paths in our testing file and randomly select 10
        # image paths
        print("[INFO] loading up test image paths...")

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")

    # load the model from disk
    if emergency_stop:
        print("[INFO] emergency stop: skipping model inference.")
        return

    unet = UNet().to(DEVICE)
    model_path = os.path.join(BASE_OUTPUT, "unet.pth")
    unet.load_state_dict(torch.load(model_path))

    unet.print_summary()

    pipeline_config = load_pipeline_config()

    if pipeline_config["inference"]["legacy_mode"]:

        # lost filenames inside IMAGE_DATASET_PATH
        file_names = [f for f in os.listdir(IMAGE_DATASET_PATH) if os.path.isfile(os.path.join(IMAGE_DATASET_PATH, f))]
        print(f"Found {len(file_names)} files in {IMAGE_DATASET_PATH}.")

        for i in range(10):
            # choose a random file
            file = np.random.choice(file_names)
            print(f"Using {file} for prediction.")

            # make predictions and visualize the results
            make_predictions(unet, os.path.join(IMAGE_DATASET_PATH, file))

    else:

        # create a mask for the s2_date
        s2_date = pipeline_config["inference"]["s2_date"]
        limit_patches = pipeline_config["inference"]["limit_patches"]

        patch_creator = RandomPatchCreator(dates=[s2_date], coverage_mode=True)
        patch_creator.open_date(s2_date)

        profile = _get_profile(s2_date)

        predicted_mask_full_tile = np.zeros((NUM_CLASSES, profile["height"], profile["width"]), dtype='float32')
        gaussian_smoother = __smooth_kern(IMAGE_SIZE)
        # copy NUM_CLASSES times the gaussian_smoother
        gaussian_smoother = np.repeat(gaussian_smoother[np.newaxis, ...], NUM_CLASSES, axis=0)

        pbar = tqdm.tqdm(range(limit_patches))
        for _ in pbar:
            # choose a random patch
            (x, y), image, mask = patch_creator.next(get_coordinates=True)
            w, h = mask.shape

            # prepare image
            image = image.transpose(2, 0, 1)
            image = image[np.newaxis, ...]

            pbar.set_description(f"Using {(x, y)} for prediction.")

            # make predictions and visualize the results
            # thus we can reuse the mask generation code
            # turn off gradient tracking
            with torch.no_grad():
                image = torch.from_numpy(image).to(DEVICE)
                _, predicted_mask = unet(image)
                predicted_mask = predicted_mask.cpu().numpy()
                # predicted_mask = np.ones((1, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), dtype='float32')

            # save the patch at the corresponding coordinates
            # we use the gaussian to smooth out the mask
            predicted_mask_full_tile[:, x:x + w, y:y + h] += predicted_mask[0, :, :, :] * gaussian_smoother

        # Create the empty JP2 file
        path = BASE_OUTPUT + f"/{s2_date}_mask_prediction.jp2"
        with rasterio.open(path, 'w', **profile) as mask_file:
            mask_file.write(get_encoded_prediction(predicted_mask_full_tile), 1)


def get_encoded_prediction(predicted_mask):
    print(f"Min/Max of predicted_mask: {np.min(predicted_mask)}/{np.max(predicted_mask)}")
    predicted_mask[:, :, :] = (predicted_mask[:, :, :] > THRESHOLD).astype(int)
    return np.argmax(predicted_mask, axis=0)


def _get_base_path(date):
    EXTRACTED_RAW_DATA = os.environ['EXTRACTED_RAW_DATA']

    folders = os.listdir(EXTRACTED_RAW_DATA)
    folders = [f for f in folders if f"_MSIL1C_{date}" in f]
    folder = folders[0]

    base_path = f"{EXTRACTED_RAW_DATA}/{folder}/GRANULE/"
    sub_folder = os.listdir(base_path)
    base_path += sub_folder[0] + '/IMG_DATA'

    return base_path


def __smooth_kern(kernel_size=IMAGE_SIZE):
    """

    """

    heaviside_lambda = lambda x: -np.sign(x) * x + 1
    heaviside_lambda_2D = lambda x, y: heaviside_lambda(x) * heaviside_lambda(y)

    xs = np.linspace(-1, 1, kernel_size)
    ys = np.linspace(-1, 1, kernel_size)
    xv, yv = np.meshgrid(xs, ys)

    return heaviside_lambda_2D(xv, yv)


def _get_profile(s2_date):
    base_path = _get_base_path(s2_date)
    B02 = f"T32TNS_{s2_date}_B02.jp2"
    B02 = rasterio.open(f"{base_path}/{B02}")

    profile = {
        'driver': 'GTiff',
        'dtype': np.uint8,
        'nodata': 0,
        'width': B02.width,
        'height': B02.height,
        'count': 1,
        'crs': B02.crs,
        'transform': B02.transform,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': 'lzw',
    }

    return profile


if __name__ == "__main__":
    main()
