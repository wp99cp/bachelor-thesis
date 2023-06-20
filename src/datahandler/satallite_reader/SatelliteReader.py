import abc
import os.path

import rasterio
from rasterio.enums import Resampling





class SatelliteReader(abc.ABC):

    def read_bands(self, tile_id: str, date: str, resolution: int):
      pass

    def get_profile(self, tile_id: str, date: str, resolution: int):
      pass