import os

import numpy as np
import rasterio


def _get_base_path(date):
    EXTRACTED_RAW_DATA = os.environ['EXTRACTED_RAW_DATA']

    folders = os.listdir(EXTRACTED_RAW_DATA)
    folders = [f for f in folders if f"_MSIL1C_{date}" in f]
    folder = folders[0]

    base_path = f"{EXTRACTED_RAW_DATA}/{folder}/GRANULE/"
    sub_folder = os.listdir(base_path)
    base_path += sub_folder[0] + '/IMG_DATA'

    return base_path


def get_profile(s2_date):
    base_path = _get_base_path(s2_date)
    B02 = f"T{os.environ['TILE_NAME']}_{s2_date}_B02.jp2"
    B02 = rasterio.open(f"{base_path}/{B02}")

    profile = {
        'driver': 'GTiff',
        'dtype': np.uint8,
        'nodata': 0,
        'width': B02.width,
        'height': B02.height,
        'count': 1,
        'crs': B02.crs,
        'transform': B02.transform,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': 'lzw',
    }

    return profile
