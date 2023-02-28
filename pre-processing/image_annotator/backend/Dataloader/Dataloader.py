import os

import cv2
import numpy as np
import rasterio
from IPython import get_ipython
from matplotlib import cm, pyplot as plt
from rasterio.windows import Window
from s2cloudless import S2PixelCloudDetector
from scipy import ndimage

DATA_DIR = '/projects/bachelor-thesis/data'
TMP_DIR = '/projects/bachelor-thesis/tmp'
MASKS_DIR = f"{DATA_DIR}/masks"

DESCRIPTIONS = ("Coastal aerosol", "Blue", "Green", "Red", "Vegetation Red Edge 1", "Vegetation Red Edge 2",
                "Vegetation Red Edge 3", "NIR", "Vegetation Red Edge 4", "Water vapour", "SWIR - Cirrus",
                "SWIR 1", "SWIR 2")

MSK_CLDPRB_20m = "MSK_CLDPRB_20m.jp2"
MSK_CLDPRB_60m = "MSK_CLDPRB_60m.jp2"
MSK_SNWPRB_20m = "MSK_SNWPRB_20m.jp2"
MSK_SNWPRB_60m = "MSK_SNWPRB_60m.jp2"


def _clip_percentile(img, low=5, high=95):
    for i in range(img.shape[0]):
        band = img[i, :, :]
        band = np.clip(band, np.percentile(band, low), np.percentile(band, high))
        band = (band - np.min(band)) / (np.max(band) - np.min(band))
        img[i, :, :] = band

    return img


class Dataloader:
    available_dates = []
    current_date = None
    profiles = None
    bands = tuple([None] * 13)

    def __init__(self):

        # get date out of filename
        self.window = None
        products_path = os.listdir(f"{DATA_DIR}/raw_data_32TNS_1C")

        # get all available dates
        self.available_dates = [product.split("_")[2] for product in products_path]

        # Check if tmp dir exists
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)

        self.change_current_date(self.available_dates[1])

        # As we are looking at the cloud probability map, we can set the threshold to 0.0
        self.cloud_detector = S2PixelCloudDetector(threshold=0, average_over=4, dilation_size=0, all_bands=True)

    def change_current_date(self, date):
        assert date in self.available_dates, "Date not available"
        self.current_date = date

        print(f"Date changed to {self.current_date}")

        self._load_bands()
        self._load_mask_coverage()

    def get_scenes_next_scenes(self):

        window = self._find_uncovered_square()
        self._load_bands_windowed(window)
        self.window = window

        scene_cloud_masks = self._compute_s2cloud_masks()

        # Colorize the cloud map using "viridis" colormap
        col_map = cm.get_cmap('viridis', 256)
        arr_cmap = col_map(scene_cloud_masks[0])
        scene_cloud_masks = np.transpose(arr_cmap[:, :, :3], (2, 0, 1))
        print("Finished loading scene_cloud_masks. Shape:", arr_cmap.shape)

        # false color RGB with band B02, B03, B12
        false_color = self.bands_windowed[0, :, :, [11, 2, 1]]
        false_color = _clip_percentile(false_color)

        # true color RGB with band B04, B03,0 B02
        true_color = self.bands_windowed[0, :, :, [3, 2, 1]]
        true_color = _clip_percentile(true_color)

        highlights = _clip_percentile(self.tci_windowed.copy(), low=0, high=100)

        tci_windowed = self.tci_windowed.copy()
        tci_windowed = tci_windowed / 255.0

        return [true_color, tci_windowed, false_color, scene_cloud_masks, highlights, self.tci_windowed]

    def _load_bands_windowed(self, window):
        bands_windowed = [None] * 13

        for i, band in enumerate(self.bands):
            # window is defined using pixel index over 10m resolution
            window_adjusted = [int(x * 10.0 / band.res[0]) for x in window]
            print(window_adjusted)
            cropped_img = band.read(window=Window(*window_adjusted))

            # Scale up to 10m resolution (512x512) using INTER_LINEAR
            if band.res[0] != 10:
                scaled_img = cv2.resize(cropped_img[0], (512, 512), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_img = cropped_img[0]
            bands_windowed[i] = scaled_img

        self.tci_windowed = self.tci.read(window=Window(*window))

        # Transform to 0-1 range
        bands_windowed = np.transpose(bands_windowed, (1, 2, 0))
        bands_windowed = np.expand_dims(bands_windowed, axis=0)

        # Normalize every band to 0-0 range
        bands_windowed = bands_windowed.astype(np.float32)

        for i in range(13):
            band = bands_windowed[0, :, :, i]

            band = band / 10000.0
            bands_windowed[0, :, :, i] = band

        self.bands_windowed = bands_windowed
        print("Finished loading bands_windowed. Shape:", self.bands_windowed.shape)

    def _compute_s2cloud_masks(self):

        bands_windowed = self.bands_windowed.copy()

        # Compute cloud probabilities
        return self.cloud_detector.get_cloud_probability_maps(bands_windowed)

    def _find_uncovered_square(self):
        # create a binary matrix to represent the area
        area = np.zeros(self.shape, dtype=bool)

        border = 2048

        if self.mask_coverage is None or len(self.mask_coverage) == 0:
            return [border, border, 512, 512]

        # mark all covered positions as True in the area matrix
        for rect in self.mask_coverage:
            x, y, w, h = rect
            area[x:x + w, y:y + h] = True

        # mark border with width of 2048 pixels as covered
        area[0:border, :] = True
        area[:, 0:border] = True
        area[-border:, :] = True
        area[:, -border:] = True

        # find the first position that is not covered by any rectangle
        uncovered_pos = np.where(area == False)

        print("Uncovered pos:", uncovered_pos)

        # return a square of size 512x512 centered on the uncovered position
        x, y = uncovered_pos[0][0], uncovered_pos[1][0]

        window = [x + int((512 - w) / 2), y + int((512 - h) / 2), 512, 512]

        # if necessary, shift the window such that is lies within the shape
        if window[0] < 0:
            window[0] = 0
        if window[1] < 0:
            window[1] = 0
        if window[0] + window[2] > self.shape[0]:
            window[0] = self.shape[0] - window[2]
        if window[1] + window[3] > self.shape[1]:
            window[1] = self.shape[1] - window[3]

        return window

    def _load_mask_coverage(self):

        print("Loading mask coverage...")
        mask_dir = f"{MASKS_DIR}/{self.current_date}"

        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
            self._create_empty_mask(f"{mask_dir}/mask_coverage.jp2")
            self._create_empty_mask(f"{mask_dir}/mask.jp2")

        with rasterio.open(f"{mask_dir}/mask_coverage.jp2") as src:
            img = src.read(1)
        labeled, n_comps = ndimage.label(img)

        # create a list to store the windows
        pixel_windows = []

        # loop through the connected components
        for i in range(1, n_comps + 1):
            # find the bounding box of the component
            rows, cols = np.where(labeled == i)
            y1, x1 = rows.min(), cols.min()
            y2, x2 = rows.max(), cols.max()

            # create a window from the bounding box
            window = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
            pixel_windows.append(window)

        # return the list of windows
        self.mask_coverage = pixel_windows

    def _create_empty_mask(self, path):

        print("Creating empty mask...")

        # Create mask_coverage for a new date
        height = self.profiles["10m"]["height"]
        width = self.profiles["10m"]["width"]

        # Set up the options for creating the empty JP2 file
        profile = {
            'driver': 'JP2OpenJPEG',
            'dtype': np.uint8,
            'nodata': 0,
            'width': width,
            'height': height,
            'count': 1,
            'crs': self.profiles["10m"]["crs"],
            'transform': self.profiles["10m"]["transform"],
            'blockxsize': 512,
            'blockysize': 512,
        }

        # Create the empty JP2 file
        with rasterio.open(path, 'w', **profile) as mask_coverage:
            mask_coverage.write(np.zeros((1, height, width), dtype='uint8'))

    def _load_bands(self):
        print("Loading bands for date: ", self.current_date)

        # Search for a folder starting with "S2B_MSIL1C_$DATE"
        folders = os.listdir(TMP_DIR)
        folders = [f for f in folders if f.startswith(f"S2B_MSIL1C_{self.current_date}")]
        folder = folders[0]

        base_path = f"{TMP_DIR}/{folder}/GRANULE/"

        # list all subfolders of the base path
        sub_folder = os.listdir(base_path)
        base_path += '/' + sub_folder[0] + '/IMG_DATA'

        # Sentinel L1C Bands
        B01 = f"T32TNS_{self.current_date}_B01.jp2"
        B02 = f"T32TNS_{self.current_date}_B02.jp2"
        B03 = f"T32TNS_{self.current_date}_B03.jp2"
        B04 = f"T32TNS_{self.current_date}_B04.jp2"
        B05 = f"T32TNS_{self.current_date}_B05.jp2"
        B06 = f"T32TNS_{self.current_date}_B06.jp2"
        B07 = f"T32TNS_{self.current_date}_B07.jp2"
        B08 = f"T32TNS_{self.current_date}_B08.jp2"
        B8A = f"T32TNS_{self.current_date}_B8A.jp2"
        B09 = f"T32TNS_{self.current_date}_B09.jp2"
        B10 = f"T32TNS_{self.current_date}_B10.jp2"
        B11 = f"T32TNS_{self.current_date}_B11.jp2"
        B12 = f"T32TNS_{self.current_date}_B12.jp2"

        bands_files = (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12)
        bands_files = (f"{base_path}/{band}" for band in bands_files)
        self.bands = tuple(rasterio.open(band) for band in bands_files)

        # TCI = True Color Image
        TCI = f"T32TNS_{self.current_date}_TCI.jp2"
        self.tci = rasterio.open(f"{base_path}/{TCI}")

        self.profiles = {
            '10m': {
                "crs": self.bands[1].crs,
                "transform": self.bands[1].transform,
                "height": self.bands[1].height,
                "width": self.bands[1].width
            },
            '20m': {
                "crs": self.bands[4].crs,
                "transform": self.bands[4].transform,
                "height": self.bands[4].height,
                "width": self.bands[4].width
            },
            '60m': {
                "crs": self.bands[0].crs,
                "transform": self.bands[0].transform,
                "height": self.bands[0].height,
                "width": self.bands[0].width
            }
        }

        self.shape = (self.bands[1].height, self.bands[1].width)
        print("Loaded bands complete")

    def _load_data_set(self):
        assert len(self.available_dates) > 0, "No dates available"

        # unzip all the data
        for date in self.available_dates:
            get_ipython().system(
                'unzip "$DATA_DIR/raw_data_32TNS_1C/"$(ls "$DATA_DIR/raw_data_32TNS_1C" | grep $date) -d $TMP_DIR')
            get_ipython().system(
                'unzip "$DATA_DIR/raw_data_32TNS_2A/"$(ls "$DATA_DIR/raw_data_32TNS_2A" | grep $date) -d $TMP_DIR')

    def update_mask(self, imgdata):
        mask_dir = f"{MASKS_DIR}/{self.current_date}"
        with rasterio.open(f"{mask_dir}/mask.jp2", 'r+') as mask:
            filename = TMP_DIR + "/mask.png"
            with open(filename, 'wb') as f:
                f.write(imgdata)

            mask_img = plt.imread(filename)

            # convert to numpy array
            mask_img = np.array(mask_img)
            mask_img = mask_img[:, :, 0]
            mask_img = cv2.resize(mask_img, (512, 512), interpolation=cv2.INTER_LINEAR)

            # convert to 0-255 range
            mask_img = mask_img * 255
            mask_img = mask_img.astype('uint8')
            np.clip(mask_img, 0, 255, out=mask_img)

            print("Min/Max: ", np.min(mask_img), np.max(mask_img))

            # Write to mask
            mask.write(mask_img, window=Window(*self.window), indexes=1)

            # print mask using matplotlib
            plt.imshow(mask.read(1))
            plt.title("Mask of " + self.current_date)
            plt.savefig(f"{mask_dir}/mask.png")

    def mark_scene_as_done(self):
        print("Loading mask coverage...")
        self.mask_coverage.append(self.window)

        mask_dir = f"{MASKS_DIR}/{self.current_date}"

        with rasterio.open(f"{mask_dir}/mask_coverage.jp2", 'r+') as mask:
            mask.write(np.ones((1, self.window[2], self.window[3]), dtype='uint8'),
                       window=Window(*self.window))

            # print mask using matplotlib
            plt.imshow(mask.read(1))
            plt.title("Mask Coverage of " + self.current_date)
            plt.savefig(f"{mask_dir}/mask_coverage.png")
