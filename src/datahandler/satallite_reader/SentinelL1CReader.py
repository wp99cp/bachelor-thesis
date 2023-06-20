import os
from enum import Enum

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.datahandler.WindowGenerator import WindowGenerator
from src.datahandler.satallite_reader.SatelliteReader import SatelliteReader


class Bands(Enum):
    B01 = ("B01", 0)
    B02 = ("B02", 1)
    B03 = ("B03", 2)
    B04 = ("B04", 3)
    B05 = ("B05", 4)
    B06 = ("B06", 5)
    B07 = ("B07", 6)
    B08 = ("B08", 7)
    B8A = ("B8A", 8)
    B09 = ("B09", 9)
    B10 = ("B10", 10)
    B11 = ("B11", 11)
    B12 = ("B12", 12)


def _get_path(tile_id: str, date: str):
    assert 'DATA_SENTINEL2' in os.environ, "Environment variable DATA_SENTINEL2 not set"
    assert tile_id is not "", "Tile id not set"
    assert date is not "", "Date not set"

    base_path = os.path.join(
        os.environ['DATA_SENTINEL2'],
        'raw_data',
        f"T{tile_id}"
    )

    assert os.path.exists(base_path), f"Path {base_path} does not exist"
    files = os.listdir(base_path)

    # find file for date
    file_for_date = [f for f in files if f"MSIL1C_{date}" in f if f.endswith('.zip')]

    assert len(file_for_date) > 0, f"No file for date {date} found"
    assert len(file_for_date) == 1, f"Multiple files found for date {date}, needs manual check: {file_for_date}"

    return os.path.join(base_path, file_for_date[0])


class SentinelL1CReader(SatelliteReader):

    def get_profile(self, tile_id: str, date: str, resolution: int):
        assert resolution == 10, "Resolutions other than 10 are not supported!"

        scene_file = _get_path(tile_id, date)

        with rasterio.open(scene_file) as s2_l1c:
            sub_datasets = s2_l1c.subdatasets

        with rasterio.open(sub_datasets[0]) as b10m:
            return {
                'driver': 'GTiff',
                'dtype': np.uint8,
                'nodata': 0,
                'width': b10m.width,
                'height': b10m.height,
                'count': 1,
                'crs': b10m.crs,
                'transform': b10m.transform,
                'blockxsize': 512,
                'blockysize': 512,
                'compress': 'lzw',
            }

    def read_bands(self, tile_id: str, date: str, resolution: int):
        """

        Reads all the bands from the zip file

        """

        assert resolution == 10, "Resolutions other than 10 are not supported!"

        scene_file = _get_path(tile_id, date)

        with rasterio.open(scene_file) as s2_l1c:
            sub_datasets = s2_l1c.subdatasets

        with rasterio.open(sub_datasets[0], dtype='uint16') as b10m:
            window_generator = WindowGenerator(b10m.transform)
            window = window_generator.get_window(tile_id=tile_id)

            description = b10m.descriptions
            assert 'B4' in description[0] and 'B3' in description[1] and 'B2' in description[2] \
                   and 'B8' in description[3], f"Bands are not in the correct order: {description}"

            [b04, b03, b02, b08] = b10m.read(
                out_shape=(
                    b10m.count,
                    int(window.height),
                    int(window.width)
                ),
                window=window,
                resampling=Resampling.nearest)

        upscale_factor = 2
        with rasterio.open(sub_datasets[1], dtype='uint16') as b20m:
            window_generator = WindowGenerator(b20m.transform)
            window = window_generator.get_window(tile_id=tile_id)

            description = b20m.descriptions
            assert 'B5' in description[0] and 'B6' in description[1] and 'B7' in description[2] \
                   and 'B8A' in description[3] and 'B11' in description[4] and 'B12' in description[5], \
                f"Bands are not in the correct order: {description}"

            # resample data to target shape
            [b05, b06, b07, b8A, b11, b12] = b20m.read(
                out_shape=(
                    b20m.count,
                    int(window.height * upscale_factor),
                    int(window.width * upscale_factor)
                ),
                window=window,
                resampling=Resampling.nearest
            )

        upscale_factor = 6
        with rasterio.open(sub_datasets[2], dtype='uint16') as b60m:
            window_generator = WindowGenerator(b60m.transform)
            window = window_generator.get_window(tile_id=tile_id)

            description = b60m.descriptions
            assert 'B1' in description[0] and 'B9' in description[1] and 'B10' in description[2], \
                f"Bands are not in the correct order: {description}"

            [b01, b09, b10] = b60m.read(
                out_shape=(
                    b60m.count,
                    int(window.height * upscale_factor),
                    int(window.width * upscale_factor)
                ),
                window=window,
                resampling=Resampling.nearest
            )

        bands = np.stack([b01, b02, b03, b04, b05, b06, b07, b08, b8A, b09, b10, b11, b12], axis=0)
        return bands.astype(np.uint16)
