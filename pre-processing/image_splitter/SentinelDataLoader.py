import json
import math
import os
import random
from multiprocessing import Lock
from typing import Tuple

import cv2
import numpy as np
import rasterio
from numpy import float32
from pytictac import ClassTimer, accumulate_time, ClassContextTimer

from PixelClassCounter import PixelClassCounter
from SentinelMemoryManager import SentinelMemoryManager
from config import NUM_ENCODED_CHANNELS, IMAGE_SIZE, MAKS_PATH, \
    EXTRACTED_RAW_DATA, SELECTED_BANDS, BORDER_WIDTH, SIGMA_CLIPPING, SIGMA_SCALE, RESULTS, LEGACY_MODE, \
    PERCENTILE_CLPPING_DYNAMIC_WORLD_METHOD, MEAN_PERCENTILES_30s, MEAN_PERCENTILES_70s, MEAN_MEANs


class SentinelDataLoader:

    def __init__(
            self,
            dates: list[str],
            mask_base_dir: str = MAKS_PATH,
            raw_data_base_dir: str = EXTRACTED_RAW_DATA,
            selected_bands: list[str] = SELECTED_BANDS,
            coverage_mode: bool = False,
            border_width: int = BORDER_WIDTH,
            timer_enabled: bool = False
    ):
        """

        Creates random patches from the raw data.

        :param dates: dates for which the patches should be created
        """

        super().__init__()

        self.mask_base_dir = mask_base_dir
        self.raw_data_base_dir = raw_data_base_dir
        self.selected_bands = selected_bands

        self.__dates = dates
        assert self.__check_date_validity() is True, "Invalid dates given."

        self.__centers = {}

        self.__memory_Manager = SentinelMemoryManager()
        self.__lock = Lock()

        self.__pixel_counter = PixelClassCounter(NUM_ENCODED_CHANNELS)

        # Coverage Mode covers the whole tile, if set to false, the patches get sampled randomly
        self.__coverage_mode = coverage_mode
        self.__border_width = border_width
        self.__coverage_coords = [self.__border_width, self.__border_width]

        self.cct = ClassTimer(objects=[self], names=["SentinelDataLoader"], enabled=timer_enabled)

    def get_bands(self, date: str):
        return self.__memory_Manager.get_date_data(date)['bands']

    def get_mask(self, date: str):
        return self.__memory_Manager.get_date_data(date)['mask'][0]  # 0 is the mask, 1 is the mask_coverage

    def get_coverage(self, date: str):
        return self.__memory_Manager.get_date_data(date)['data_coverage']

    def get_mask_coverage(self, date: str):
        return self.__memory_Manager.get_date_data(date)['mask'][1]  # 0 is the mask, 1 is the mask_coverage

    def get_max_number_of_cover_patches(self, date: str):
        """
        Returns the maximum number of patches that can be created for the given date if the coverage mode is enabled.

        :param date: the date
        :return: the maximum number of patches

        if the date is not opened, this function throws an error
        if the coverage mode is not enabled, this function throws an error

        """

        assert self.__coverage_mode is True, "Coverage mode is not enabled."
        assert date in self.__centers.keys(), "Date not opened."

        shape = (10980, 10980)
        border_margin = self.__border_width * 2

        return math.floor((shape[0] - border_margin) / (IMAGE_SIZE // 2)) * \
            math.ceil((shape[1] - border_margin) / (IMAGE_SIZE // 2))

    @accumulate_time
    def open_date(self, date: str, fast_mode: bool = False):
        """
        Sets the date for which the patches should be created.

        :param date: the date
        :param fast_mode: if true, the bands are not loaded into memory
        """

        assert date in self.__dates, "Invalid date given."

        if date not in self.__centers.keys():
            self.__centers[date] = []

        # lock the memory manager
        self.__lock.acquire()
        if not self.__memory_Manager.has_date(date):
            self.__memory_Manager.add_date(date)

            self.__lock.release()

            if not fast_mode:
                bands, coverage = self.__load_bands_into_memory(date)
            else:
                bands = None
                coverage = None

            self.__memory_Manager.add_date_data(date, {
                'bands': bands,
                'mask': self.__load_mask_into_memory(date),
                'data_coverage': coverage
            })

        else:
            self.__lock.release()

    def close_date(self, date: str):
        """
        Closes the date used to save memory.

        :param date: the date
        """

        self.__memory_Manager.close_date(date)

    def get_number_of_dates(self):
        """
        Returns the number of dates.

        :return: the number of dates
        """

        return len(self.__dates)

    def get_PixelClassCounter(self):
        return self.__pixel_counter

    @accumulate_time
    def __load_mask_into_memory(self, date: str):
        print("    Loading mask and mask_coverage...")

        # check if file exists self.mask_base_dir + '/' + _date
        if os.path.isfile(self.mask_base_dir + '/' + date + '/mask_coverage.jp2') and \
                os.path.isfile(self.mask_base_dir + '/' + date + '/mask.jp2'):
            mask = self.__load_mask(date)
            mask_coverage = self.__load_mask_coverage(date)

            mask_data = mask.read(1)
            mask_coverage_data = mask_coverage.read(1)
            print(f"    Mask and mask_coverage loaded for {date}.")
            return mask_data, mask_coverage_data

        return None, None

    @accumulate_time
    def __load_bands_into_memory(self, date):

        """
        Loads the bands into memory.
        All the bands are resized 10m resolution and normalized to 0-255.

        :param date: the date
        """

        # Plot the mask_coverage and the mask
        print(f"    Loading bands for {date} into memory...")

        band_file_names = {
            'B01': f"T{os.environ['TILE_NAME']}_{date}_B01.jp2",
            'B02': f"T{os.environ['TILE_NAME']}_{date}_B02.jp2",
            'B03': f"T{os.environ['TILE_NAME']}_{date}_B03.jp2",
            'B04': f"T{os.environ['TILE_NAME']}_{date}_B04.jp2",
            'B05': f"T{os.environ['TILE_NAME']}_{date}_B05.jp2",
            'B06': f"T{os.environ['TILE_NAME']}_{date}_B06.jp2",
            'B07': f"T{os.environ['TILE_NAME']}_{date}_B07.jp2",
            'B08': f"T{os.environ['TILE_NAME']}_{date}_B08.jp2",
            'B8A': f"T{os.environ['TILE_NAME']}_{date}_B8A.jp2",
            'B09': f"T{os.environ['TILE_NAME']}_{date}_B09.jp2",
            'B10': f"T{os.environ['TILE_NAME']}_{date}_B10.jp2",
            'B11': f"T{os.environ['TILE_NAME']}_{date}_B11.jp2",
            'B12': f"T{os.environ['TILE_NAME']}_{date}_B12.jp2",
            # 'TCI': f"T{os.environ['TILE_NAME']}_{date}_TCI.jp2", # the TCI can be reconstructed from bands 4, 3, 2
        }

        # Search for a folder starting with "S2B_MSIL1C_$DATE"
        base_dir = self.__get_date_base_dir(date)
        band_files = [f"{base_dir}/{band_file_names[b]}" for b in self.selected_bands if b in band_file_names.keys()]

        # add additional metadata to the bands 32TNS_30m_DEM_AW3D30.tif
        elevation_tiff = f"{self.raw_data_base_dir}/{os.environ['TILE_NAME']}_auxiliary_data/{os.environ['TILE_NAME']}_30m_DEM_AW3D30.tif"

        # upscale all bands to 10m resolution using cv2.resize
        b02_meta = rasterio.open(f"{base_dir}/{band_file_names['B02']}").meta
        image_width = b02_meta['width']
        image_height = b02_meta['height']

        assert image_width == image_height == 10980, "Invalid image size."

        summary_stats = []

        # open and pre-process all bands (except additional metadata, i.g. elevation)
        coverage = np.zeros((image_height, image_width), dtype=np.uint8)
        bands_data = []
        for i, band in enumerate(band_files):

            # load band into memory
            with ClassContextTimer(parent_obj=self, block_name="rasterio.open",
                                   parent_method_name="__load_bands_into_memory"):
                with rasterio.open(band, dtype='uint16') as b:
                    band_data = b.read(1)

                    if band_data.shape == (image_height, image_width):
                        coverage |= (band_data > 0)

            with ClassContextTimer(parent_obj=self, block_name="pre_process_band",
                                   parent_method_name="__load_bands_into_memory"):

                mean = np.mean(band_data)
                sigma = np.std(band_data)

                percentile_01 = np.percentile(band_data, 1)
                percentile_1 = np.percentile(band_data, 1)
                percentile_5 = np.percentile(band_data, 5)
                percentile_10 = np.percentile(band_data, 10)
                percentile_30 = np.percentile(band_data, 30)
                percentile_70 = np.percentile(band_data, 70)
                percentile_90 = np.percentile(band_data, 90)
                percentile_95 = np.percentile(band_data, 95)
                percentile_99 = np.percentile(band_data, 99)
                percentile_999 = np.percentile(band_data, 99.9)

                min_val_raw, max_val_raw = np.min(band_data), np.max(band_data)

                # resize image
                band_data = cv2.resize(band_data, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
                min_val_resized, max_val_resized = np.min(band_data), np.max(band_data)

                # Clipping to overall channel [mean - SIGMA_SCALE * sigma range, mean + SIGMA_SCALE * sigma range]
                if SIGMA_CLIPPING:
                    lower_bound = max(0, mean - SIGMA_SCALE * sigma)
                    upper_bound = min(mean + SIGMA_SCALE * sigma, 65_535)
                    np.clip(band_data, lower_bound, upper_bound, out=band_data)

                min_val_clipped, max_val_clipped = np.min(band_data), np.max(band_data)

                # Convert to float32
                band_data = band_data.astype(np.float32)

                # Then normalizing according to (val - min) / (max - min)
                # Then rescaling to 0-255
                if SIGMA_CLIPPING:
                    band_data = (band_data - np.min(band_data)) / (np.max(band_data) - np.min(band_data)) * 255
                elif PERCENTILE_CLPPING_DYNAMIC_WORLD_METHOD:

                    # log scale the raw data (map 0 to 0)
                    np.log10(band_data, where=band_data > 0, out=band_data, dtype=float32)

                    # print(f"Min/Max (log10):\t{np.min(band_data):2.4f}/{np.max(band_data):2.4f}")

                    # squeeze the data to 15% and 85% of [0, 1]
                    # val_out = (val_in - c) * (b - a) / (d - c) + a
                    a = -1.7346  # 0.15 == 1 / (1 + Exp[-x]) // Solve
                    b = +1.7346  # 0.85 == 1 / (1 + Exp[-x]) // Solve
                    c = np.log10(MEAN_PERCENTILES_30s[i])
                    d = np.log10(MEAN_PERCENTILES_70s[i])

                    band_data = (band_data - c) * (b - a) / (d - c) + a
                    # print(f"Min/Max (normalized):\t{np.min(band_data):2.4f}/{np.max(band_data):2.4f}")

                    # sigmoid function to normalize the data to 0-1
                    band_data = 1 / (1 + np.exp(-band_data))
                    # print(f"Min/Max (sigmoid):\t{np.min(band_data):2.4f}/{np.max(band_data):2.4f}")

                    # shift mean to, the final values should be in the range of [-1, 1] and has a max range of 1
                    offset = np.log10(MEAN_MEANs[i])
                    offset = (offset - c) * (b - a) / (d - c) + a
                    offset = 1 / (1 + np.exp(-offset))
                    band_data = band_data - offset

                    # print(f"Min/Max (centralized):\t{np.min(band_data):2.4f}/{np.max(band_data):2.4f}")
                    # print()

                else:
                    if not LEGACY_MODE:
                        # Clip values to 0-10_000
                        np.clip(band_data, 0, 10_000, out=band_data)
                    band_data = band_data / 10_000 * 255

                # save band data
                bands_data.append(band_data)

                # ########################
                # Report some statistics
                # ########################

                band_id = i + 1
                if i == 8:
                    band_id = "8A"
                elif i > 8:
                    band_id = i

                summary_stats.append({
                    "band": band_id,
                    "raw": {
                        "min": min_val_raw,
                        "max": max_val_raw,
                        "percentile_01": percentile_01,
                        "percentile_1": percentile_1,
                        "percentile_5": percentile_5,
                        "percentile_10": percentile_10,
                        "percentile_30": percentile_30,
                        "percentile_70": percentile_70,
                        "percentile_90": percentile_90,
                        "percentile_95": percentile_95,
                        "percentile_99": percentile_99,
                        "percentile_999": percentile_999,
                        "sigma": sigma,
                        "mean": mean,
                    },
                    "resized": {
                        "min": min_val_resized,
                        "max": max_val_resized,
                    },
                    "clipped": {
                        "min": min_val_clipped,
                        "max": max_val_clipped,
                    }
                })

        # we load the elevation data separately, since it used a different data type
        if "ELEV" in self.selected_bands:
            with rasterio.open(elevation_tiff, dtype='float32') as b:
                band_data = b.read(1)

                print(f"Size of elevation data: {band_data.shape}")

                mean = np.mean(band_data)
                sigma = np.std(band_data)

                percentile_5 = np.percentile(band_data, 5)
                percentile_95 = np.percentile(band_data, 95)

                min_val_raw, max_val_raw = np.min(band_data), np.max(band_data)

                # normalize elevation data
                band_data = cv2.resize(band_data, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
                # convert to float32
                band_data = band_data.astype(np.float32)
                print(f"Size of elevation data: {band_data.shape}")
                print(f"Datatype of elevation data: {band_data.dtype}")

                min_val_resized, max_val_resized = np.min(band_data), np.max(band_data)

                if LEGACY_MODE:
                    band_data = band_data / 10_000 * 255
                else:
                    # based on the 32TNS data
                    MAX_ELEV = 3913.0
                    MEAN_ELEV = 1862.78
                    band_data = (band_data - MEAN_ELEV) / MAX_ELEV

                print(f"Elevation Data range: {np.min(band_data):2.4f}/{np.max(band_data):2.4f}")
                print(f"Elevation Data Summary Stats: {np.mean(band_data):2.4f}/{np.std(band_data):2.4f}")

                bands_data.append(band_data)

                # ########################
                # Report some statistics
                # ########################

                summary_stats.append({
                    "band": "ELEV",
                    "raw": {
                        "min": min_val_raw,
                        "max": max_val_raw,
                        "percentile_5": percentile_5,
                        "percentile_95": percentile_95,
                        "sigma": sigma,
                        "mean": mean,
                    },
                    "resized": {
                        "min": min_val_resized,
                        "max": max_val_resized,
                    },
                    "clipped": {
                        "min": min_val_resized,
                        "max": max_val_resized,
                    }
                })

        class NpEncoder(json.JSONEncoder):
            """
            Special json encoder for numpy types
            Source: https://stackoverflow.com/a/57915246
            """

            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        # save summary statistics
        if not os.path.exists(f"{RESULTS}/summary_stats"):
            os.makedirs(f"{RESULTS}/summary_stats")

        with open(f"{RESULTS}/summary_stats/{date}_{os.environ['TILE_NAME']}.json", 'w') as f:
            json.dump(summary_stats, f, indent=4, cls=NpEncoder)

        # convert to numpy array
        bands_data = np.array(bands_data)

        print(f"  » Memory usage: {(bands_data.nbytes / (1024 ** 3)):.2f} GB")
        print(f"    Opened {len(bands_data)} bands for date {date}.")
        print(f"    Number of bands: {bands_data.shape}")

        return bands_data, coverage

    def __get_date_base_dir(self, date):
        folders = [folder for folder in os.listdir(self.raw_data_base_dir) if f"_MSIL1C_{date}" in folder]
        folder = folders[0]
        base_dir = f"{self.raw_data_base_dir}/{folder}/GRANULE"
        sub_folders = os.listdir(base_dir)
        base_dir += '/' + sub_folders[0] + '/IMG_DATA'
        return base_dir

    def __check_date_validity(self):
        """
        Checks if the given dates are valid.

        This function checks if the mask for all given dates exists.
        And if the raw data for all given dates exists.

        :return: True if all dates are valid, else False
        """

        # TODO: implement this function
        return True

    def next(self, get_coordinates: bool = False, date: str = None):
        """
        Creates a random patch from the raw data.
        :param get_coordinates: if True, the coordinates of the patch are returned
        :return: the patch, its center coordinates, and the mask
        """

        # get a random date from the open dates
        assert len(self.__memory_Manager.get_open_dates()) > 0, "No dates are open, please open at least one date."
        if date is None:
            date = random.choice(list(self.__memory_Manager.get_open_dates()))

        # get a random patch from the given date
        if not self.__coverage_mode:
            return self.random_patch(date, get_coordinates)

        # get location
        coords = self.__next_coverage(date)
        return self.__get_patch(date, get_coordinates, coords)

    def __next_coverage(self, date):

        self.__coverage_coords[0] = self.__coverage_coords[0] + IMAGE_SIZE // 2
        width = self.get_bands(date).shape[2] - self.__border_width
        height = self.get_bands(date).shape[1] - self.__border_width

        if self.__coverage_coords[0] >= width:
            self.__coverage_coords[0] = self.__border_width
            self.__coverage_coords[1] = self.__coverage_coords[1] + IMAGE_SIZE // 2

        if self.__coverage_coords[1] >= height:
            raise StopIteration

        return self.__coverage_coords

    def random_patch(self, date: str, get_coordinates: bool = False):
        """
        Creates a random patch from the raw data.

        :return: the patch, its center coordinates, and the mask
        """

        # sample new location
        mask_coverage = self.get_mask_coverage(date)
        coords = self.__sample_patch_center_coords(mask_coverage)

        return self.__get_patch(date, get_coordinates, coords)

    def __get_patch(self, date: str, get_coordinates: bool = False, coords: Tuple[int, int] = None):
        if date not in self.__centers.keys():
            self.__centers[date] = []

        self.__centers[date].append(coords)

        x, y = coords

        # load image patch
        bands = self.get_bands(date)
        img_patch = bands[:, x:x + IMAGE_SIZE, y:y + IMAGE_SIZE]
        img_patch = np.moveaxis(img_patch, 0, -1)

        # load mask patch
        mask = self.get_mask(date)

        mask_patch = None
        if mask is not None:
            mask_patch = mask[x:x + IMAGE_SIZE, y:y + IMAGE_SIZE]
            mask_patch = mask_patch * (255 / NUM_ENCODED_CHANNELS)

            # used to count the number of pixels in each class
            self.__pixel_counter.update(mask_patch, date)
            mask_patch = mask_patch.astype(np.uint8)

        if get_coordinates:
            return coords, img_patch, mask_patch

        return img_patch, mask_patch

    def get_patch_centers(self, date):

        assert date in self.__centers, f"Date {date} not found in centers."
        return self.__centers[date]

    # sample a random patch from the mask_coverage
    # where the mask_coverage is 1
    def __sample_patch_center_coords(self, _mask_coverage, count=0):

        assert count < 1_000, "Could not find a valid patch after 1'000 tries."

        # TODO: add option to oversample water pixels

        # sample a random patch
        _x = np.random.randint(0, _mask_coverage.shape[0] - IMAGE_SIZE)
        _y = np.random.randint(0, _mask_coverage.shape[1] - IMAGE_SIZE)

        # check if the patch is valid
        if np.sum(_mask_coverage[_x:_x + IMAGE_SIZE, _y:_y + IMAGE_SIZE]) == IMAGE_SIZE ** 2:
            return _x, _y
        else:
            return self.__sample_patch_center_coords(_mask_coverage, count + 1)

    # Load the mask_coverage and the mask
    def __load_mask_coverage(self, _date):
        mask_coverage_path = self.mask_base_dir + '/' + _date + '/mask_coverage.jp2'
        return rasterio.open(mask_coverage_path)

    # Load the mask
    def __load_mask(self, _date):
        mask_path = self.mask_base_dir + '/' + _date + '/mask.jp2'
        return rasterio.open(mask_path)

    def report_time(self):
        print(self.cct.__str__())
