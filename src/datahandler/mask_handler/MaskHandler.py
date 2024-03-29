import os
from enum import Enum

import numpy as np
import rasterio

from src.datahandler.WindowGenerator import WindowGenerator


class Masks(Enum):
    """

    Mask Types:
    Maps the mask type to the folder name in which the masks are stored.

    """

    AUTO_GENERATED = "autogenerated_masks"
    CLEANED = "cleaned_masks"


def _get_base_path(tile_id: str, date: str, mask_type: Masks = Masks.AUTO_GENERATED):

    assert os.environ['DATA_SENTINEL2'] is not None, "Environment variable DATA_SENTINEL2 must be set"
    assert tile_id is not None, "Tile id must be set"
    assert date is not None, "Date must be set"
    assert mask_type is not None, "Mask type must be set"

    base_path = os.path.join(
        os.environ['DATA_SENTINEL2'],
        'masks',
        f"T{tile_id}",
        mask_type.value,
        date
    )

    return base_path


class MaskHandler:

    def __init__(self):
        self.profile = None

    def load_data(self, tile_id: str, date: str, resolution: int = 10, mask_type: Masks = Masks.AUTO_GENERATED):
        assert resolution == 10, "Only 10m resolution is supported for masks"

        self.profile = None  # set profile

        base_path = _get_base_path(tile_id, date, mask_type)

        mask_path = os.path.join(base_path, f"mask.jp2")
        coverage_path = os.path.join(base_path, f"coverage.jp2")

        # assert that the files exists
        assert os.path.exists(mask_path), f"Mask file {mask_path} does not exist"
        assert os.path.exists(coverage_path), f"Coverage file {coverage_path} does not exist"

        with rasterio.open(mask_path) as file:
            window_generator = WindowGenerator(file.transform)
            window = window_generator.get_window(tile_id=tile_id)
            mask = file.read(1, window=window)

        with rasterio.open(coverage_path) as file:
            window_generator = WindowGenerator(file.transform)
            window = window_generator.get_window(tile_id=tile_id)
            coverage = file.read(1, window=window)

        return mask, coverage

    def save_data(
            self,
            mask_type: Masks,
            tile_id: str,
            date: str,
            data: np.ndarray,
            resolution: int = 10,
            profile: rasterio.profiles.Profile = None
    ):
        """
        Saves the mask data to the correct folder.


        """

        assert resolution == 10, "Only 10m resolution is supported for masks"

        if profile is None:
            profile = self.profile

        assert profile is not None, "Profile must be set before saving data"

        base_path = _get_base_path(tile_id, date, mask_type)

        # create directory if not exists
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        mask_path = os.path.join(base_path, f"mask.jp2")
        coverage_path = os.path.join(base_path, f"coverage.jp2")

        for i, path in enumerate([mask_path, coverage_path]):
            with rasterio.open(path, 'w', **profile) as file:
                window_generator = WindowGenerator(file.transform)
                window = window_generator.get_window(tile_id=tile_id)
                file.write(data[i], 1, window=window)
