import os
import uuid

import cv2
import numpy as np
import rasterio
from IPython import get_ipython
from PIL import Image
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

img_src = f"{TMP_DIR}/res/imgs"
if not os.path.exists(img_src):
    os.makedirs(img_src)


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
        self.refs = {}
        products_path = os.listdir(f"{DATA_DIR}/raw_data_32TNS_1C")

        # get all available dates
        self.available_dates = [product.split("_")[2] for product in products_path]

        # Check if tmp dir exists
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)

        # Those days are selected in a way to have a good mix of snow and cloud coverage
        # Images are distributed over several months to have a good mix of seasons
        selected_dates = ["20211227T102339", "20210720T101559", "20210908T101559", "20210819T101559", "20211018T101939"]

        # randomly set a date
        self.change_current_date("20211008T101829") # (np.random.choice(selected_dates))

        # As we are looking at the cloud probability map, we can set the threshold to 0.0
        self.cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)

    def change_current_date(self, date):
        assert date in self.available_dates, "Date not available"
        self.current_date = date

        print(f"Date changed to {self.current_date}")

        self._load_bands()
        self._load_mask_coverage()

    def generate_next(self):

        ref = str(uuid.uuid1())

        window = self._find_uncovered_square()
        self._load_bands_windowed(window)

        scene_cloud_probs, cloud_mask04, cloud_mask06 = self._compute_s2cloud_masks()

        # false color RGB with band B02, B03, B12
        false_color = self.bands_windowed[0, :, :, [11, 2, 1]]
        false_color = _clip_percentile(false_color)

        # true color RGB with band B04, B03,0 B02
        true_color = self.bands_windowed[0, :, :, [3, 2, 1]]
        true_color = _clip_percentile(true_color)

        highlights = _clip_percentile(self.tci_windowed.copy(), low=0, high=100)

        tci_windowed = self.tci_windowed.copy()
        tci_windowed = tci_windowed / 255.0

        exoLab_classifications = self.exolabs_classification.read(window=Window(*window))
        exoLab_snow_classifications = exoLab_classifications[0, :, :]
        exoLab_classifications = exoLab_classifications[1, :, :]

        # map and convert to RGB
        cmap = plt.cm.get_cmap('nipy_spectral', np.max(exoLab_snow_classifications) + 1)
        exoLab_snow_classifications = np.array(cmap(exoLab_snow_classifications))
        exoLab_snow_classifications = exoLab_snow_classifications[:, :, :3]
        exoLab_snow_classifications = exoLab_snow_classifications.transpose(2, 0, 1)

        # map and convert to RGB
        cmap = plt.cm.get_cmap('nipy_spectral', np.max(exoLab_classifications) + 1)
        exoLab_classifications = np.array(cmap(exoLab_classifications))
        exoLab_classifications = exoLab_classifications[:, :, :3]
        exoLab_classifications = exoLab_classifications.transpose(2, 0, 1)

        # glacier and surface water (both are at 30m resolution)
        window_60m = window.copy()
        window_60m = (window_60m[0] // 3, window_60m[1] // 3, window_60m[2] // 3, window_60m[3] // 3)
        surface_water_mask = self.exolabs_classification.read(2, window=Window(*window_60m))

        # upscale to 512x512 using cv2
        surface_water_mask = cv2.resize(surface_water_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # map and convert to RGB
        cmap = plt.cm.get_cmap('nipy_spectral', np.max(surface_water_mask) + 1)
        surface_water_mask = np.array(cmap(surface_water_mask))
        surface_water_mask = surface_water_mask[:, :, :3]
        surface_water_mask = surface_water_mask.transpose(2, 0, 1)

        scenes = [
            true_color,
            tci_windowed,
            false_color,
            scene_cloud_probs,
            highlights,
            self.tci_windowed,
            cloud_mask04,
            cloud_mask06,
            exoLab_snow_classifications,
            exoLab_classifications,
            surface_water_mask
        ]

        # Generate PNGs
        scene_names = [f"scene_original" if i == 0 else f"scene_{i - 1}" for i in range(len(scenes))]
        for i, scene in enumerate(scenes):
            img = Image.fromarray(np.uint8(scene * 255).transpose(1, 2, 0))
            img.save(f'{img_src}/{ref}_{scene_names[i]}.png')
            scene_names[i] = f'{ref}_{scene_names[i]}.png'

        self.refs[ref] = {
            "current_date": self.current_date,
            "window": window,
            "scenes": scene_names
        }

        print("\n\n=======================================\n\n")

        return ref

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

        scene_cloud_probs = self.cloud_detector.get_cloud_probability_maps(bands_windowed)

        # Colorize the cloud map using "viridis" colormap
        col_map = cm.get_cmap('viridis', 256)
        arr_cmap = col_map(scene_cloud_probs[0])
        scene_cloud_probs = np.transpose(arr_cmap[:, :, :3], (2, 0, 1))

        cloud_mask_04 = self.cloud_detector.get_mask_from_prob(scene_cloud_probs, threshold=0.4)
        cloud_mask_04 = np.stack([cloud_mask_04[0], cloud_mask_04[0], cloud_mask_04[0]], axis=0)

        cloud_mask_06 = self.cloud_detector.get_mask_from_prob(scene_cloud_probs, threshold=0.6)
        cloud_mask_06 = np.stack([cloud_mask_06[0], cloud_mask_06[0], cloud_mask_06[0]], axis=0)

        # Compute cloud probabilities
        return scene_cloud_probs, cloud_mask_04, cloud_mask_06

    def _find_uncovered_square(self):

        # create a binary matrix to represent the area
        area = np.zeros(self.shape, dtype=bool)

        border = 2048

        assert self.mask_coverage is not None, "Mask coverage not loaded"
        if len(self.mask_coverage) == 0:
            self.mask_coverage = [[border, border, 512, 512]]
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

        # add the window to the list of covered areas
        assert len(window) == 4, "Window must be a tuple of length 4"
        self.mask_coverage.append(tuple(np.array(window)))

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
            'driver': 'GTiff',
            'dtype': np.uint8,
            'nodata': 0,
            'width': width,
            'height': height,
            'count': 1,
            'crs': self.profiles["10m"]["crs"],
            'transform': self.profiles["10m"]["transform"],
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'lzw',
        }

        # Create the empty JP2 file
        with rasterio.open(path, 'w', **profile) as mask:
            mask.write(np.zeros((1, height, width), dtype='uint8'))

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

        # ==========
        # open ExoLabs classification
        # ==========
        ExoLabs_path = f"{TMP_DIR}/ExoLabs_classification_S2/"

        # convert date to ExoLabs format, i.g. from "20210710T101559" to 2021-07-10
        date = self.current_date[:4] + '-' + self.current_date[4:6] + '-' + self.current_date[6:8]

        # search for the ExoLabs classification file
        ExoLabs_files = os.listdir(ExoLabs_path)
        ExoLabs_file = [f for f in ExoLabs_files if f.startswith(f"S2_32TNS_{date}")]
        self.exolabs_classification = rasterio.open(f"{TMP_DIR}/ExoLabs_classification_S2/{ExoLabs_file[0]}")

        # ==========
        # open auxiliary data
        # ==========

        surfaceWater = f"{TMP_DIR}/32TNS_auxiliary_data/32TNS_30m_JRC_surfaceWater.tif"
        self.surfaceWater = rasterio.open(surfaceWater)

        glacier = f"{TMP_DIR}/32TNS_auxiliary_data/32TNS_30m_Glacier_RGIv6.tif"
        self.glacier = rasterio.open(glacier)

        print("Loaded bands complete")

    def _load_data_set(self):
        assert len(self.available_dates) > 0, "No dates available"

        # unzip all the data
        for date in self.available_dates:
            get_ipython().system(
                'unzip "$DATA_DIR/raw_data_32TNS_1C/"$(ls "$DATA_DIR/raw_data_32TNS_1C" | grep $date) -d $TMP_DIR')
            get_ipython().system(
                'unzip "$DATA_DIR/raw_data_32TNS_2A/"$(ls "$DATA_DIR/raw_data_32TNS_2A" | grep $date) -d $TMP_DIR')

        get_ipython().system('unzip "$DATA_DIR/ExoLabs_classification_S2.zip -d $TMP_DIR')
        get_ipython().system('unzip "$DATA_DIR/32TNS_auxiliary_data.zip -d $TMP_DIR')

    def update_mask(self, ref, mask_img):

        window = self.refs[ref]["window"]
        current_date = self.refs[ref]["current_date"]

        mask_dir = f"{MASKS_DIR}/{current_date}"
        with rasterio.open(f"{mask_dir}/mask.jp2", 'r+') as mask:
            # convert to numpy array
            mask_img = np.array(mask_img)
            mask_img = mask_img.astype('uint8')

            # reshape to 2D array
            mask_img = mask_img.reshape((512, 512))
            np.clip(mask_img, 0, 255, out=mask_img)

            # plot the mask using matplotlib and the verdis palette
            plt.figure(figsize=(20, 20))
            plt.imshow(mask_img, cmap=plt.cm.get_cmap('viridis', np.max(mask_img) + 1))
            plt.savefig(f"{mask_dir}/mask.png")

            print("Min/Max: ", np.min(mask_img), np.max(mask_img))

            # Write to mask
            mask.write(mask_img, window=Window(*window), indexes=1)

    def mark_scene_as_done(self, ref):

        window = self.refs[ref]["window"]
        current_date = self.refs[ref]["current_date"]

        print("Loading mask coverage...")
        mask_dir = f"{MASKS_DIR}/{current_date}"

        with rasterio.open(f"{mask_dir}/mask_coverage.jp2", 'r+') as mask:
            mask.write(np.ones((1, window[2], window[3]), dtype='uint8'),
                       window=Window(*window))
