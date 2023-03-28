import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as tsfm

from DataLoader.SegmentationDataset import SegmentationDataset
from augmentation.Augmentation import Augmentation
from augmentation.ChannelDropout import ChannelDropout
from augmentation.HorizontalFlip import HorizontalFlip
from augmentation.RandomErasing import RandomErasing
from augmentation.VerticalFlip import VerticalFlip
from configs.config import report_config, IMAGE_DATASET_PATH, MASK_DATASET_PATH, TEST_SPLIT, BATCH_SIZE, PIN_MEMORY, \
    DEVICE, BASE_OUTPUT, ENABLE_DATA_AUGMENTATION, IMAGE_FLIP_PROB, CHANNEL_DROPOUT_PROB, \
    COVERED_PATCH_SIZE_MIN, COVERED_PATCH_SIZE_MAX, LIMIT_DATASET_SIZE, NUM_DATA_LOADER_WORKERS, BATCH_PREFETCHING
from model.Model import UNet
from model.inference import make_predictions
from training import train_unet


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

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(image_paths, mask_paths, test_size=TEST_SPLIT, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # define transformations
    transforms = tsfm.Compose([tsfm.ToTensor()])
    augmentations: list[Augmentation] = [
        HorizontalFlip(prob=IMAGE_FLIP_PROB),
        VerticalFlip(prob=IMAGE_FLIP_PROB),
        ChannelDropout(prob=CHANNEL_DROPOUT_PROB),
        RandomErasing(prob=CHANNEL_DROPOUT_PROB, min_size=COVERED_PATCH_SIZE_MIN, max_size=COVERED_PATCH_SIZE_MAX)
    ]

    # create the train and test datasets
    train_ds = SegmentationDataset(image_paths=trainImages, mask_paths=trainMasks, transforms=transforms,
                                   apply_augmentations=ENABLE_DATA_AUGMENTATION, augmentations=augmentations)
    test_ds = SegmentationDataset(image_paths=testImages, mask_paths=testMasks,
                                  transforms=transforms, apply_augmentations=False)

    # create the training and test data loaders
    num_workers = min(os.cpu_count(), 6)  # max 6 workers
    num_workers = NUM_DATA_LOADER_WORKERS if NUM_DATA_LOADER_WORKERS > 0 else num_workers
    print(f"\n[INFO] using {num_workers} workers to load the data.")
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, prefetch_factor=BATCH_PREFETCHING,
                              pin_memory=PIN_MEMORY, num_workers=num_workers)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, prefetch_factor=BATCH_PREFETCHING,
                             pin_memory=PIN_MEMORY, num_workers=num_workers)

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

    # lost filenames inside IMAGE_DATASET_PATH
    file_names = [f for f in os.listdir(IMAGE_DATASET_PATH) if os.path.isfile(os.path.join(IMAGE_DATASET_PATH, f))]
    print(f"Found {len(file_names)} files in {IMAGE_DATASET_PATH}.")

    for i in range(10):
        # choose a random file
        file = np.random.choice(file_names)
        print(f"Using {file} for prediction.")

        # make predictions and visualize the results
        make_predictions(unet, os.path.join(IMAGE_DATASET_PATH, file))


if __name__ == "__main__":
    main()
