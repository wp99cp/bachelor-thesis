import os

SAMPLES_PER_DATE = 8192

IMAGE_SIZE = 256
NUM_ENCODED_CHANNELS = 5

# TODO: add the modified band B11
SELECTED_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "ELEV"]

# ====================================
# ====================================
# Loader Config (Normalisation, etc.)
# ====================================
# ====================================

# disable sigma clipping and normalization for
# inference with algorithms older than (including) `e40b271`.
# and enable the legacy mode
LEGACY_MODE = False

SIGMA_CLIPPING = True
SIGMA_SCALE = 2.0

# val = (val - min) / (max - min)
NORMALIZE = True

# ====================================
# ====================================
# Automatic Configs
# ====================================
# ====================================
DATA_DIR = os.environ['DATA_DIR']
EXTRACTED_RAW_DATA = os.environ['EXTRACTED_RAW_DATA']
MAKS_PATH = os.environ['ANNOTATED_MASKS_DIR']
DATASET_DIR = os.environ['DATASET_DIR']
RESULTS = os.environ['RESULTS_DIR']

BORDER_WIDTH = 256


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
