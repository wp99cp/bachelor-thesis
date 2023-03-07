import os

import torch

# base path of the dataset
BASE_DIR = '/projects/bachelor-thesis/tmp'
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
BASE_OUTPUT = "res"

# define the test split:
# the fraction of the dataset we will keep aside for the test set
TEST_SPLIT = 0.15

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 5  # B02, B03, B04, B08, B08
NUM_CLASSES = 4  # we have background, snow and clouds
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 25
BATCH_SIZE = 64

# define the input image dimensions
IMAGE_SIZE = 128

# define threshold to filter weak predictions
THRESHOLD = 0.4

# Number of channels used to encode the grayscale image
NUM_ENCODED_CHANNELS = 5

# class weights for background, snow, clouds, water
CLASS_WEIGHTS = [0.62062, 0.12658, 0.24463, 0.00816]

# ====================================
# ====================================
# Automatically determined parameters
# ====================================
# ====================================

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False


# ====================================
# ====================================
# Some helper functions
# ====================================
# ====================================

def report_config():
    """Report the configuration of the model."""
    print("\nConfiguration:")
    print(f"- Using device: {DEVICE}")
    print(f"- Using {NUM_CHANNELS} channels")
    print(f"- Using {NUM_CLASSES} classes")
    print(f"- Using {NUM_LEVELS} levels")
    print(f"- Using {NUM_EPOCHS} epochs")
    print(f"- Using {BATCH_SIZE} batch size")
    print(f"- Using {IMAGE_SIZE} image size")
    print(f"- Using {THRESHOLD} threshold")
    print(f"- Using {NUM_ENCODED_CHANNELS} encoded channels")
    print(f"- Using {INIT_LR} initial learning rate")
    print(f"- Using {TEST_SPLIT} test split")
    print(f"- Using {DATASET_PATH} dataset path")
    print(f"- Using {IMAGE_DATASET_PATH} image dataset path")
    print(f"- Using {MASK_DATASET_PATH} mask dataset path")
    print(f"- Using {PIN_MEMORY} pin memory")
    print("\n")
