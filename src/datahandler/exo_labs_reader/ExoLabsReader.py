import os

import rasterio
from rasterio.enums import Resampling

from src.datahandler.WindowGenerator import WindowGenerator


def _get_path(tile_id: str, date: str):
    base_path = os.path.join(
        os.environ['DATA_SENTINEL2'],
        'exoLabs',
        f"T{tile_id}"
    )

    files = os.listdir(base_path)

    date_converted = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"

    # find file for date
    file_for_date = [f for f in files if f"S2_{tile_id}_{date_converted}" in f and f.endswith(".tif")]

    assert len(file_for_date) > 0, f"No file for date {date_converted} found"
    assert len(
        file_for_date) == 1, f"Multiple files found for date {date_converted}, needs manual check: {file_for_date}"

    return os.path.join(base_path, file_for_date[0])


class ExoLabsReader:

    def __init__(self):
        pass

    def load_data(self, tile_id: str, date: str, resolution: int = 10):
        assert resolution == 10, "Only 10m resolution is supported for ExoLabs data"

        path = _get_path(tile_id, date)

        print(f"Loading ExoLabs data from {path}")
        with rasterio.open(path) as file:
            window_generator = WindowGenerator(file.transform)
            window = window_generator.get_window(tile_id=tile_id)

            return file.read(
                1,
                out_shape=(1, int(window.height), int(window.width)),
                window=window,
                resampling=Resampling.nearest
            )
