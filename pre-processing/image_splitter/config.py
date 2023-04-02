import os

SAMPLES_PER_DATE = 8192

IMAGE_SIZE = 256
NUM_ENCODED_CHANNELS = 5
SELECTED_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "ELEV"]

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

BORDER_WITH = 2048


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
