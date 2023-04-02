import os
import random
from multiprocessing import Lock
from typing import Tuple

import cv2
import numpy as np
import rasterio
from numpy import float32

from PixelClassCounter import PixelClassCounter
from SentinelMemoryManager import SentinelMemoryManager
from config import NUM_ENCODED_CHANNELS, IMAGE_SIZE, MAKS_PATH, \
    EXTRACTED_RAW_DATA, SELECTED_BANDS, BORDER_WITH


class RandomPatchCreator:

    def __init__(
            self,
            dates: list[str],
            mask_base_dir: str = MAKS_PATH,
            raw_data_base_dir: str = EXTRACTED_RAW_DATA,
            selected_bands: list[str] = SELECTED_BANDS,
            coverage_mode: bool = False
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
        self.__coverage_coords = [BORDER_WITH, BORDER_WITH]

    def get_bands(self, date: str):
        return self.__memory_Manager.get_date_data(date)['bands']

    def get_mask(self, date: str):
        return self.__memory_Manager.get_date_data(date)['mask'][0]  # 0 is the mask, 1 is the mask_coverage

    def get_mask_coverage(self, date: str):
        return self.__memory_Manager.get_date_data(date)['mask'][1]  # 0 is the mask, 1 is the mask_coverage

    def open_date(self, date: str):
        """
        Sets the date for which the patches should be created.

        :param date: the date
        """

        assert date in self.__dates, "Invalid date given."

        if date not in self.__centers.keys():
            self.__centers[date] = []

        # lock the memory manager
        self.__lock.acquire()
        if not self.__memory_Manager.has_date(date):
            self.__memory_Manager.add_date(date)

            self.__lock.release()

            self.__memory_Manager.add_date_data(date, {
                'bands': self.__load_bands_into_memory(date),
                'mask': self.__load_mask_into_memory(date),
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

    def __load_mask_into_memory(self, date: str):
        print("    Loading mask and mask_coverage...")

        mask = self.__load_mask(date)
        mask_coverage = self.__load_mask_coverage(date)

        mask_data = mask.read(1)
        mask_coverage_data = mask_coverage.read(1)
        print(f"    Mask and mask_coverage loaded for {date}.")

        return mask_data, mask_coverage_data

    def __load_bands_into_memory(self, date):
        # Plot the mask_coverage and the mask

        print(f"    Loading bands for {date} into memory...")

        band_file_names = {
            'B01': f"T32TNS_{date}_B01.jp2",
            'B02': f"T32TNS_{date}_B02.jp2",
            'B03': f"T32TNS_{date}_B03.jp2",
            'B04': f"T32TNS_{date}_B04.jp2",
            'B05': f"T32TNS_{date}_B05.jp2",
            'B06': f"T32TNS_{date}_B06.jp2",
            'B07': f"T32TNS_{date}_B07.jp2",
            'B08': f"T32TNS_{date}_B08.jp2",
            'B8A': f"T32TNS_{date}_B8A.jp2",
            'B09': f"T32TNS_{date}_B09.jp2",
            'B10': f"T32TNS_{date}_B10.jp2",
            'B11': f"T32TNS_{date}_B11.jp2",
            'B12': f"T32TNS_{date}_B12.jp2",
            'TCI': f"T32TNS_{date}_TCI.jp2",
        }

        # Search for a folder starting with "S2B_MSIL1C_$DATE"
        base_dir = self.__get_date_base_dir(date)
        band_files = [f"{base_dir}/{band_file_names[b]}" for b in self.selected_bands]

        # open all the bands
        bands_data = []

        for band in band_files:
            with rasterio.open(band, dtype='uint16') as b:
                bands_data.append(b.read(1))

        # upscale all bands to 10m resolution using cv2.resize
        b02_meta = rasterio.open(f"{base_dir}/{band_file_names['B02']}").meta

        image_with = b02_meta['width']
        image_height = b02_meta['height']
        bands_data = np.array(
            [cv2.resize(band, (image_with, image_height), interpolation=cv2.INTER_CUBIC).astype(float32) for band in
             bands_data]
        )

        print(f"    » Memory usage: {(bands_data.nbytes / (1024 ** 3)):.2f} GB")
        print(f"    Opened {len(bands_data)} bands for date {date}.")
        print(f"    Number of bands: {bands_data.shape}")

        # normalize the bands
        bands_data = bands_data / 10_000 * 255

        return bands_data

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

    def next(self, get_coordinates: bool = False):
        """
        Creates a random patch from the raw data.
        :param get_coordinates: if True, the coordinates of the patch are returned
        :return: the patch, its center coordinates, and the mask
        """

        # get a random date from the open dates
        assert len(self.__memory_Manager.get_open_dates()) > 0, "No dates are open, please open at least one date."
        date = random.choice(list(self.__memory_Manager.get_open_dates()))

        # get a random patch from the given date
        if not self.__coverage_mode:
            return self.random_patch(date, get_coordinates)

        # get location
        coords = self.__next_coverage(date)
        return self.__get_patch(date, get_coordinates, coords)

    def __next_coverage(self, date):

        self.__coverage_coords[0] = self.__coverage_coords[0] + IMAGE_SIZE // 2
        width = self.get_bands(date).shape[2] - BORDER_WITH
        height = self.get_bands(date).shape[1] - BORDER_WITH

        if self.__coverage_coords[0] >= width:
            self.__coverage_coords[0] = BORDER_WITH
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

        # load mask patch
        mask = self.get_mask(date)
        mask_patch = mask[x:x + IMAGE_SIZE, y:y + IMAGE_SIZE]
        mask_patch = mask_patch * (255 / NUM_ENCODED_CHANNELS)

        # used to count the number of pixels in each class
        self.__pixel_counter.update(mask_patch, date)

        # load image patch
        bands = self.get_bands(date)
        img_patch = bands[:, x:x + IMAGE_SIZE, y:y + IMAGE_SIZE]

        # normalize the image patch to [0, 1]
        img_patch = img_patch.astype(np.float32)
        img_patch = img_patch / 255.0
        img_patch = np.moveaxis(img_patch, 0, -1)

        if get_coordinates:
            return coords, img_patch, mask_patch.astype(np.uint8)

        return img_patch, mask_patch.astype(np.uint8)

    def get_patch_centers(self, date):

        assert date in self.__centers, f"Date {date} not found in centers."
        return self.__centers[date]

    # sample a random patch from the mask_coverage
    # where the mask_coverage is 1
    def __sample_patch_center_coords(self, _mask_coverage, count=0):

        assert count < 1_000, "Could not find a valid patch after 1'000 tries."

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
