import os

import torch

# define the test split:
# the fraction of the dataset we will keep aside for the test set
TEST_SPLIT = 0.15

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
IMAGE_SIZE = 128  # defines the input image dimensions
NUM_CHANNELS = 13  # all satellite images have 13 channels

# Maks Settings
NUM_CLASSES = 4
CLASS_NAMES = ["snow", "clouds", "water", "thin_clouds"]
NUM_ENCODED_CHANNELS = 5  # Number of channels used to encode the grayscale image
CLASS_WEIGHTS = [0.25052, 0.00214, 0.01381, 0.02479]  # class weights for snow, clouds, water

# define threshold to filter weak predictions
THRESHOLD = 0.5

# ====================================
# ====================================
# Training Hyperparameters
# ====================================
# ====================================

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
MOMENTUM = 0.95
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 25
BATCH_SIZE = 100

# ====================================
# ====================================
# Data Augmentation Settings
# ====================================
# ====================================

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
COVERED_PATCH_SIZE_MIN = 16  # in pixels
COVERED_PATCH_SIZE_MAX = 32  # in pixels

# ====================================
# ====================================
# Automatically determined parameters
# ====================================
# ====================================

# base path of the dataset
BASE_DIR = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else '/projects/bachelor-thesis/tmp'
DATASET_PATH = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else os.path.join(BASE_DIR, "dataset")

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
