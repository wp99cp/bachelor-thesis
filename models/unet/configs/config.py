import os

import torch

# define the test split:
# the fraction of the dataset we will keep aside for the test set
TEST_SPLIT = 0.15

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
IMAGE_SIZE = 256  # defines the input image dimensions
NUM_CHANNELS = 13  # all satellite images have 13 channels

# Maks Settings
NUM_CLASSES = 4
CLASS_NAMES = ["background", "snow", "clouds", "water"]  # "thin_clouds"
NUM_ENCODED_CHANNELS = 5  # Number of channels used to encode the grayscale image
CLASS_WEIGHTS = [0.40922, 0.29371, 0.28889, 0.00818]  # class weights for background, snow, clouds, water

# define threshold to filter weak predictions
THRESHOLD = 0.65

# 0 for unlimited
# artificially limit the number of samples in the dataset
# by only using the first LIMIT_DATASET_SIZE samples (paths)
LIMIT_DATASET_SIZE = 0

# ====================================
# ====================================
# Training Hyperparameters
# ====================================
# ====================================

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
MOMENTUM = 0.950
WEIGHT_DECAY = 0.150
NUM_EPOCHS = 100
BATCH_SIZE = 24

# ====================================
# ====================================
# Data Augmentation Settings
# ====================================
# ====================================

# 0 - uses max 3 workers, 1 - uses 1 worker, >1 - uses the specified number of workers
# should be below 6
NUM_DATA_LOADER_WORKERS = 4
BATCH_PREFETCHING = 64
BATCH_MIXTURE = 24

# Data Augmentation Settings
ENABLE_DATA_AUGMENTATION = True

# the best value according to the paper is 0.3
# "MixChannel: Advanced Augmentation for Multi spectral Satellite Images" (https://www.mdpi.com/2072-4292/13/11/2181)
# Every channel gets dropped with a probability of 0.3
CHANNEL_DROPOUT_PROB = 0.1

# probability of flipping the image horizontally and/or vertically (this happens independently)
IMAGE_FLIP_PROB = 0.3

# cover a random patch of the image (i.g. setting all channels and the mask to zero)
PATCH_COVERING_PROB = 0.3
COVERED_PATCH_SIZE_MIN = 8  # in pixels
COVERED_PATCH_SIZE_MAX = 64  # in pixels

# ====================================
# ====================================
# Automatically determined parameters
# ====================================
# ====================================

# base path of the dataset
BASE_DIR = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else '/projects/bachelor-thesis/tmp'
DATASET_PATH = os.environ['DATASET_DIR'] if 'DATASET_DIR' in os.environ else os.path.join(BASE_DIR, "dataset")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
BASE_OUTPUT = os.environ['RESULTS_DIR'] if 'RESULTS_DIR' in os.environ else 'res'

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# ====================================
# ====================================
# Assertions to ensure correct configuration
# ====================================
# ====================================

assert NUM_CLASSES == len(CLASS_NAMES), "Number of classes must match number of class names."


# ====================================
# ====================================
# Some helper functions
# ====================================
# ====================================

def report_config():
    """Report the configuration of the model."""

    print(f"\nConfiguration:")
    for var in globals():

        # check if it is a variable and not a function
        if not var.startswith('__') and not callable(var) and var == var.upper():
            print(f" - {var} = {eval(var)}")

    print(f"\n\n")
