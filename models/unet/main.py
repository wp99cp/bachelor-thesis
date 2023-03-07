import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms as trsf

from DataLoader.SegmentationDataset import SegmentationDataset
from configs.config import report_config, IMAGE_DATASET_PATH, MASK_DATASET_PATH, TEST_SPLIT, BATCH_SIZE, PIN_MEMORY, \
    DEVICE, BASE_OUTPUT
from model.Model import UNet
from model.inference import make_predictions
from model.training import training


def load_data():
    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))

    print(f"[INFO] found a total of {len(imagePaths)} images in {IMAGE_DATASET_PATH}.")

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths, test_size=TEST_SPLIT, random_state=42)

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # define transformations
    transforms = trsf.Compose([
        trsf.ToPILImage(),
        # resize is not needed as we are using a pretrained model
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        trsf.ToTensor(),
        trsf.RandomVerticalFlip(p=0.5),
        trsf.RandomHorizontalFlip(p=0.5),
        trsf.RandomErasing(p=0.5, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0, inplace=False),
    ])

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=BATCH_SIZE,
                             pin_memory=PIN_MEMORY, num_workers=os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False, batch_size=BATCH_SIZE,
                            pin_memory=PIN_MEMORY, num_workers=os.cpu_count())

    print(f"[INFO] loaded {len(trainDS)} examples in the train set.")
    print(f"[INFO] loaded {len(testDS)} examples in the test set.")

    return trainLoader, testLoader, trainDS, testDS, trainImages, testImages


def print_data_sample(trainLoader):
    # show a sample image and mask
    sample = next(iter(trainLoader))

    img = sample[0][0].permute(1, 2, 0)
    masks = sample[1][0].permute(1, 2, 0)

    # use matplotlib to display the image and mask
    # show a legend for the mask next to the image
    # i.g. blue = snow, green = clouds, red = water
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title("Image")

    ax[1].imshow(masks[:, :, 0])
    ax[1].set_title("Mask")

    sample_data_path = os.path.join(BASE_OUTPUT, "sample_data.png")
    plt.savefig(sample_data_path)


def main():
    report_config()

    trainLoader, testLoader, trainDS, testDS, _, testImages = load_data()
    print_data_sample(trainLoader)

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

    # iterate over the randomly selected test image paths
    for path in imagePaths:
        # make predictions and visualize the results
        make_predictions(unet, path)


if __name__ == "__main__":
    main()
