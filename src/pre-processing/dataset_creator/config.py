import os

from src.datahandler.DataHandler import ALL_BANDS
from src.datahandler.auxiliary_reader.AuxiliaryReader import AuxiliaryData

DATASET_DIR = os.path.join(os.environ['TMP_DIR'], 'dataset')

# ====================================

SELECTED_BANDS = ALL_BANDS
AUXILIARY_DATA = [AuxiliaryData.DEM]
PATCH_SIZE = 256
NUM_ENCODED_CHANNELS = 5

# ====================================

SAMPLES_PER_DATE = 96  # 8192


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
