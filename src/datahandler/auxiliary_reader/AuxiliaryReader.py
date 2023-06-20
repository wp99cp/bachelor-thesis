import os
from enum import Enum

import rasterio
from rasterio.enums import Resampling

from src.datahandler.WindowGenerator import WindowGenerator


class AuxiliaryData(Enum):
    TREE_CANOPY = "30m_treeCanopyCover.tif"
    SURFACE_WATER = "30m_JRC_surfaceWater.tif"
    GLACIER = "30m_Glacier_RGIv6.tif"
    DEM = "30m_DEM_AW3D30.tif"


class AuxiliaryReader:

    def load_data(self, tile_id: str, auxiliary_data: AuxiliaryData):
        auxiliary_file_path = os.path.join(
            os.environ['BASE_DIR'],
            'data',
            'auxiliary_data',
            f"T{tile_id}",
            f"{tile_id}_{auxiliary_data.value}"
        )

        upscale_factor = 3
        with rasterio.open(auxiliary_file_path) as file:
            window_generator = WindowGenerator(file.transform)
            window = window_generator.get_window(tile_id=tile_id)

            return file.read(
                out_shape=(
                    file.count,
                    int(window.height * upscale_factor),
                    int(window.width * upscale_factor)
                ),
                window=window,
                resampling=Resampling.nearest
            )[0]
