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
from configs.config import report_config, IMAGE_DATASET_PATH, MASK_DATASET_PATH, TEST_SPLIT, BATCH_SIZE, PIN_MEMORY, \
    DEVICE, BASE_OUTPUT, NUM_CLASSES, CLASS_NAMES
from model.Model import UNet
from model.inference import make_predictions
from model.training import training


def load_data():
    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_files(IMAGE_DATASET_PATH, validExts=("npz",))))
    maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))

    print(f"[INFO] found a total of {len(imagePaths)} images in '{IMAGE_DATASET_PATH}'.")
    print(f"[INFO] found a total of {len(maskPaths)} images in '{MASK_DATASET_PATH}'.")

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths, test_size=TEST_SPLIT, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # define transformations
    transforms = tsfm.Compose([
        tsfm.ToTensor()
    ])

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
                                 transforms=transforms, apply_augmentations=False)

    # create the training and test data loaders
    num_workers = np.min(os.cpu_count(), 16)
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=BATCH_SIZE,
                             pin_memory=PIN_MEMORY, num_workers=num_workers)
    testLoader = DataLoader(testDS, shuffle=False, batch_size=BATCH_SIZE,
                            pin_memory=PIN_MEMORY, num_workers=num_workers)

    print(f"[INFO] loaded {len(trainDS)} examples in the train set.")
    print(f"[INFO] loaded {len(testDS)} examples in the test set.")

    return trainLoader, testLoader, trainDS, testDS, trainImages, testImages


def print_data_sample(trainLoader):
    # show a sample image and mask
    sample = next(iter(trainLoader))

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
    assert NUM_CLASSES == len(CLASS_NAMES), "Number of classes must match number of class names."
    report_config()

    # check if the flag "--retrain" is set
    # if so, train the model
    if "--retrain" in sys.argv:
        trainLoader, testLoader, trainDS, testDS, _, testImages = load_data()
        print_data_sample(trainLoader)

        print("[INFO] retraining model...")
        training(trainLoader, testLoader, trainDS, testDS)

        # load the image paths in our testing file and randomly select 10
        # image paths
        print("[INFO] loading up test image paths...")
        imagePaths = np.random.choice(testImages, size=1)

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")

    # load the model from disk
    unet = UNet().to(DEVICE)
    model_path = os.path.join(BASE_OUTPUT, "unet.pth")
    unet.load_state_dict(torch.load(model_path))

    # lost filenames inside IMAGE_DATASET_PATH
    file_names = [f for f in os.listdir(IMAGE_DATASET_PATH) if os.path.isfile(os.path.join(IMAGE_DATASET_PATH, f))]
    print(f"Found {len(file_names)} files in {IMAGE_DATASET_PATH}.")
    # choose a random file
    file = np.random.choice(file_names)
    print(f"Using {file} for prediction.")

    # make predictions and visualize the results
    make_predictions(unet, os.path.join(IMAGE_DATASET_PATH, file))


if __name__ == "__main__":
    main()
