import os
import sys

import cv2
import numpy as np
import rasterio
import yaml
from rasterio.windows import from_bounds
from s2cloudless import S2PixelCloudDetector

from config import BORDER_WIDTH

TMP_DIR = os.environ['TMPDIR']
ANNOTATED_MASKS_DIR = os.environ['ANNOTATED_MASKS_DIR']


class MaskGenerator:

    def __init__(self, sample_date):
        self.sample_date = sample_date
        self.profile = self._get_profile()

    def _get_profile(self):
        base_path = self._get_base_path()

        B02 = f"T32TNS_{self.sample_date}_B02.jp2"
        B02 = rasterio.open(f"{base_path}/{B02}")

        profiles = {
            "crs": B02.crs,
            "transform": B02.transform,
            "height": B02.height,
            "width": B02.width
        }
        self.bounds = B02.bounds
        self.bounds = (self.bounds[0] + BORDER_WIDTH, self.bounds[1] + BORDER_WIDTH, self.bounds[2] - BORDER_WIDTH,
                       self.bounds[3] - BORDER_WIDTH)
        print(f"Bounds: {self.bounds}")
        self.dimensions = B02.read(1, window=from_bounds(*self.bounds, B02.transform)).shape
        print(f"Dimensions: {self.dimensions}")
        B02.close()

        # Set up the options for creating the empty JP2 file
        return {
            'driver': 'GTiff',
            'dtype': np.uint8,
            'nodata': 0,
            'width': profiles["width"],
            'height': profiles["height"],
            'count': 1,
            'crs': profiles["crs"],
            'transform': profiles["transform"],
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'lzw',
        }

    def _get_base_path(self, date=None):

        if date is None:
            date = self.sample_date

        folders = os.listdir(TMP_DIR)
        folders = [f for f in folders if f"_MSIL1C_{date}" in f]
        folder = folders[0]

        base_path = f"{TMP_DIR}/{folder}/GRANULE/"
        sub_folder = os.listdir(base_path)
        base_path += '/' + sub_folder[0] + '/IMG_DATA'

        return base_path

    def generate_masks(self, dates):
        print(f"Creating masks for {len(dates)} dates.")

        # open general auxiliary data
        COP_Lakes_10m = f"{TMP_DIR}/32TNS_auxiliary_data/32TNS_10m_COP_Lakes.tif"
        COP_Lakes_10m = rasterio.open(COP_Lakes_10m)
        COP_Lakes_10m_arr = COP_Lakes_10m.read(1, window=from_bounds(*self.bounds, transform=COP_Lakes_10m.transform))
        COP_Lakes_10m.close()

        JRC_surfaceWater_30m = f"{TMP_DIR}/32TNS_auxiliary_data/32TNS_30m_JRC_surfaceWater.tif"
        JRC_surfaceWater_30m = rasterio.open(JRC_surfaceWater_30m)
        JRC_surfaceWater_30m_arr = JRC_surfaceWater_30m.read(1, window=from_bounds(*self.bounds,
                                                                                   transform=JRC_surfaceWater_30m.transform))
        JRC_surfaceWater_30m.close()

        # resample the water mask to 10m using cv2
        JRC_surfaceWater_10m_arr = cv2.resize(JRC_surfaceWater_30m_arr,
                                              (COP_Lakes_10m_arr.shape[1], COP_Lakes_10m_arr.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)

        Glacier_RGIv6_30m = f"{TMP_DIR}/32TNS_auxiliary_data/32TNS_30m_Glacier_RGIv6.tif"
        Glacier_RGIv6_30m = rasterio.open(Glacier_RGIv6_30m)
        Glacier_RGIv6_30m_arr = Glacier_RGIv6_30m.read(1, window=from_bounds(*self.bounds,
                                                                             transform=Glacier_RGIv6_30m.transform))
        Glacier_RGIv6_30m.close()

        # resample the glacier mask to 10m using cv2
        Glacier_RGIv6_10m_arr = cv2.resize(Glacier_RGIv6_30m_arr,
                                           (COP_Lakes_10m_arr.shape[1], COP_Lakes_10m_arr.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

        for date in dates:
            date_parsed = date.split('T')[0]
            date_parsed = date_parsed[:4] + '-' + date_parsed[4:6] + '-' + date_parsed[6:]
            print(f"Creating masks for date {date} = {date_parsed}.")

            self._create_empty_mask(date, f"{ANNOTATED_MASKS_DIR}/{date}/mask.jp2")
            self._create_empty_mask(date, f"{ANNOTATED_MASKS_DIR}/{date}/mask_coverage.jp2", value=1)

            # s2cloudless prediction
            cloud_mask = self.s2_cloudless_prediction(date)
            print(f"    Cloud mask shape: {cloud_mask.shape}")
            print(f"    Min/Max: {np.min(cloud_mask)}/{np.max(cloud_mask)}")
            print(f"    Mean: {np.mean(cloud_mask)}")

            # set water pixels
            mask = f"{ANNOTATED_MASKS_DIR}/{date}/mask.jp2"
            mask = rasterio.open(mask, 'r+')

            # get ExoLab classification
            path = f"{TMP_DIR}/ExoLabs_classification_S2/"
            files = os.listdir(path)
            files = [f for f in files if f.startswith(f"S2_32TNS_{date_parsed}")]
            file = files[0]

            # read the ExoLab classification
            exo_lab_snow_class = rasterio.open(f"{path}/{file}")
            exo_lab_snow_class_arr = exo_lab_snow_class.read(
                1, window=from_bounds(*self.bounds, transform=exo_lab_snow_class.transform))

            window = rasterio.windows.from_bounds(*self.bounds, transform=mask.transform)
            mask_arr = mask.read(1, window=window)

            mask_arr[COP_Lakes_10m_arr == 1] = 3
            mask_arr[JRC_surfaceWater_10m_arr == 1] = 3
            mask_arr[Glacier_RGIv6_10m_arr == 1] = 1
            mask_arr[exo_lab_snow_class_arr == 4] = 1
            mask_arr[cloud_mask == 1] = 2

            mask.write(mask_arr, 1, window=window)

    def s2_cloudless_prediction(self, date):

        base_path = self._get_base_path(date)

        # Sentinel L1C Bands
        B01 = f"T32TNS_{date}_B01.jp2"
        B02 = f"T32TNS_{date}_B02.jp2"
        B03 = f"T32TNS_{date}_B03.jp2"
        B04 = f"T32TNS_{date}_B04.jp2"
        B05 = f"T32TNS_{date}_B05.jp2"
        B06 = f"T32TNS_{date}_B06.jp2"
        B07 = f"T32TNS_{date}_B07.jp2"
        B08 = f"T32TNS_{date}_B08.jp2"
        B8A = f"T32TNS_{date}_B8A.jp2"
        B09 = f"T32TNS_{date}_B09.jp2"
        B10 = f"T32TNS_{date}_B10.jp2"
        B11 = f"T32TNS_{date}_B11.jp2"
        B12 = f"T32TNS_{date}_B12.jp2"

        # Read the bands
        bands_files = (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12)
        bands_files = (f"{base_path}/{band}" for band in bands_files)

        # Read the bands as arrays
        print("  - Loading the bands into memory...")
        bands_arrs = []
        for band in bands_files:
            band = rasterio.open(band)
            band_arr = band.read(1, window=from_bounds(*self.bounds, transform=band.transform))
            band.close()

            if band_arr.shape != self.dimensions:
                band_arr = cv2.resize(band_arr, self.dimensions, interpolation=cv2.INTER_NEAREST)

            band_arr = band_arr / 10_000.0
            bands_arrs.append(band_arr)

        bands_arrs = np.array(bands_arrs)
        bands_arrs = np.transpose(bands_arrs, (1, 2, 0))

        print(f"    Finished loading the bands into memory. Starting the cloud detection...")
        cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)
        scene_cloud_prob = cloud_detector.get_cloud_probability_maps(bands_arrs[np.newaxis, ...])
        cloud_mask = cloud_detector.get_mask_from_prob(scene_cloud_prob, threshold=0.4)
        print(f"    Finished the cloud detection.")

        return cloud_mask[0]

    def _create_empty_mask(self, current_date, path, value=0):

        # Create the directory if it does not exist
        os.makedirs(f"{ANNOTATED_MASKS_DIR}/{current_date}", exist_ok=True)

        # Create the empty JP2 file
        with rasterio.open(path, 'w', **self.profile) as mask:
            mask.write(np.zeros((1, self.profile["height"], self.profile["width"]), dtype='uint8'))

        # Create the empty JP2 file
        if value != 0:
            with rasterio.open(path, 'r+') as mask:
                window = rasterio.windows.from_bounds(*self.bounds, transform=mask.transform)

                # set value to value instead the window
                mask_arr = mask.read(1, window=window)
                mask_arr.fill(value)
                mask.write(mask_arr, 1, window=window)


def main():
    # get the pyth to the config file from the config_file arg
    config_file = sys.argv[2]
    print(config_file)

    config = None

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if config is None:
        print("Could not load the config file.")
        return

    mask_generator = MaskGenerator(sample_date=config['data_handling']['s2_dates'][0])
    mask_generator.generate_masks(dates=config['data_handling']['s2_dates'])


if __name__ == "__main__":
    main()
