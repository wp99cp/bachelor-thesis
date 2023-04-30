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

# disable SIGMA_CLIPPING and PERCENTILE_CLIPPING for
# inference with algorithms older than (including) `e40b271`.
# and enable the legacy mode
LEGACY_MODE = False  # legacy: True

PERCENTILE_CLPPING_DYNAMIC_WORLD_METHOD = True  # legacy: False
# used mean percentiles for each band (B1 - B12)
MEAN_PERCENTILES_30s = [1390, 1032, 759, 544, 633, 900, 974, 907, 981, 470, 16, 236, 151]
MEAN_PERCENTILES_70s = [9084, 9637, 9415, 10306, 10437, 10482, 10377, 10113, 10184, 5892, 504, 4044, 3295]
MEAN_MEANs = [3747, 3561, 3307, 3407, 3596, 4049, 4203, 4089, 4268, 2270, 138, 1831, 1332]

# this is an experimental feature and should be disabled
SIGMA_CLIPPING = False
SIGMA_SCALE = 2.0

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
# Verify the configuration
# ====================================
# ====================================

assert PERCENTILE_CLPPING_DYNAMIC_WORLD_METHOD ^ SIGMA_CLIPPING, "Only one of PERCENTILE_CLIPPING and SIGMA_CLIPPING can be enabled!"
assert len(MEAN_PERCENTILES_70s) == len(
    MEAN_PERCENTILES_30s) == 13, "MEAN_PERCENTILES_70s and MEAN_PERCENTILES_30s must be defined for each band (B1 - B12)!"
assert SIGMA_SCALE > 0, "SIGMA_SCALE must be greater than 0!"


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
