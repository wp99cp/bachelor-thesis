import os
import sys

SAMPLES_PER_DATE = 8192

IMG_SIZE = 256
NUM_ENCODED_CHANNELS = 5
SELECTED_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


# ====================================
# ====================================
# Automatic Configs
# ====================================
# ====================================
BASE_DIR = os.environ['BASE_DIR']
TMP_DIR = os.environ['TMPDIR']
MAKS_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else '/data/masks'
DATASET_DIR = os.environ['DATASET_DIR']
RESULTS = os.environ['RESULTS_DIR']