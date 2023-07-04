import os

import torch

# define the test split:
# the fraction of the dataset we will keep aside for the test set
TEST_SPLIT = 0.15

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
IMAGE_SIZE = 256  # defines the input image dimensions
NUM_CHANNELS = 14  # all satellite images have 13 channels
NUM_ENCODED_CHANNELS = 4

# Maks Settings
NUM_CLASSES = 4
CLASS_NAMES = ["background", "snow", "clouds", "water"]  # "thin_clouds"
CLASS_WEIGHTS = [0.4132, 0.31163, 0.26706, 0.00812]  # class weights for background, snow, clouds, water
ROOT_WEIGHTS = True  # instead of using the class weights, use the root weights

# define threshold to filter weak predictions
THRESHOLD = 0.90

# 0 for unlimited
# artificially limit the number of samples in the dataset
# by only using the first LIMIT_DATASET_SIZE samples (paths)
LIMIT_DATASET_SIZE = 0

# ====================================
# ====================================
# Training Hyper Parameters
# ====================================
# ====================================

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001  # if using amp the INIT_LR should be below 0.001
WEIGHT_DECAY = 0.1
NUM_EPOCHS = 60
BATCH_SIZE = 12 # 48  # fastest on Euler (assuming Quadro RTX 6000) is 32, however this may be too small (nan loss)

WEIGHT_DECAY_PLATEAU_PATIENCE = 1
EARLY_STOPPING_PATIENCE = 30

STEPS_PER_EPOCH = 4096
STEPS_PER_EPOCH_TEST = 1024

# switches to mixed precision training after the specified epoch
# if set to 0, mixed precision training is disabled
# make sure to disable mixed precision during inference!
USE_PIXED_PRECISION = False
GRADIENT_CLIPPING = True

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
CHANNEL_DROPOUT_PROB = 0.3

# probability of flipping the image horizontally and/or vertically (this happens independently)
IMAGE_FLIP_PROB = 0.25

# cover a random patch of the image (i.g. setting all channels and the mask to zero)
PATCH_COVERING_PROB = 0.4
COVERED_PATCH_SIZE_MIN = 8  # in pixels
COVERED_PATCH_SIZE_MAX = 128  # in pixels

# ====================================
# ====================================
# Inference Settings
# ====================================
# ====================================

LOAD_CORRUPT_WEIGHTS = False

# ====================================
# ====================================
# Automatically determined parameters
# ====================================
# ====================================

# base path of the dataset
BASE_DIR = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else '/projects/bachelor-thesis/tmp'
BASE_DIR = os.environ['TMP_DIR'] if 'TMP_DIR' in os.environ else '/projects/bachelor-thesis/tmp'
DATASET_PATH = os.environ['DATASET_DIR'] if 'DATASET_DIR' in os.environ else os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(BASE_DIR, "dataset") if DATASET_PATH == '' else DATASET_PATH

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
BASE_OUTPUT = os.environ['RESULTS_DIR'] if 'RESULTS_DIR' in os.environ else 'res'
AUXILIARY_DATA_DIR = os.environ['AUXILIARY_DATA_DIR']

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

CONTINUE_TRAINING = os.environ['CONTINUE_TRAINING'] == 1 if 'CONTINUE_TRAINING' in os.environ else False

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
