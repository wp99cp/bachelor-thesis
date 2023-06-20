import cv2
import numpy as np
from pytictac import ClassTimer, accumulate_time

from src.datahandler.auxiliary_reader.AuxiliaryReader import AuxiliaryReader, AuxiliaryData
from src.datahandler.exo_labs_reader.ExoLabsReader import ExoLabsReader
from src.datahandler.mask_handler.MaskHandler import MaskHandler, Masks
from src.datahandler.satallite_reader.SatelliteReader import SatelliteReader
from src.datahandler.satallite_reader.SentinelL1CReader import Bands

ALL_BANDS = [Bands.B01, Bands.B02, Bands.B03, Bands.B04, Bands.B05, Bands.B06, Bands.B07,
             Bands.B08, Bands.B8A, Bands.B09, Bands.B10, Bands.B11, Bands.B12]


class DataHandler:
    """

    The Dataloader class is responsible for loading the data from the satellite reader and converting it into a format
    that can be used by the model / pipeline.

    The DataHandler caches the data of a single scene in memory.

    @arg satelliteReader: the satellite reader to use
    @arg resolution: the resolution of the data

    """

    def __init__(self, satelliteReader: SatelliteReader, resolution: int = 10):
        self.resolution = resolution
        self.satelliteReader = satelliteReader

        self.__auxiliary_reader = AuxiliaryReader()
        self.__exo_labs_reader = ExoLabsReader()
        self.__mask_handler = MaskHandler()

        self.has_open_scene = False

        self.tile_id = None
        self.date = None

        # internal cache
        self.__bands = None
        self.__masks = None
        self.__auxiliary_data = None
        self.__exo_labs = None
        self.__profile = None
        self.__satellite_data_coverage = None

        self.__shape = None

        self.cct = ClassTimer(objects=[self], names=["DataHandler"])

    def _normalize_bands(self, bands: np.ndarray) -> np.ndarray:
        return bands

    def _normalize_auxiliary_data(self, data: np.ndarray, auxiliary_data: AuxiliaryData) -> np.ndarray:
        return data

    @accumulate_time
    def open_scene(self, tile_id: str, date: str):
        """
        Opens the scene for the given tile_id and date and saves it in the memory.

        This function cannot be called if a scene is already open. In that case,
        the scene should be closed first.

        @arg tile_id: the tile id of the scene
        @arg date: the date of the scene

        # TODO: add option to load masks, exoLabs, and auxiliary data eagerly

        """

        assert not self.has_open_scene, "A scene is already open, close it first"

        self.tile_id = tile_id
        self.date = date

        print(" » Opening scene for tile_id: {} and date: {}".format(tile_id, date))

        self.has_open_scene = True
        bands = np.stack(self.satelliteReader.read_bands(tile_id, date, self.resolution))
        self.__satellite_data_coverage = self.__get_satellite_data_coverage(bands)
        self.__bands = self._normalize_bands(bands)

        self.__profile = self.satelliteReader.get_profile(tile_id=tile_id, date=date, resolution=self.resolution)

        self.__shape = self.__bands.shape[1:]

    @accumulate_time
    def get_bands(self, bands: list[Bands] = None):
        """

        Returns the bands for the current scene.

        @arg bands: the bands to return, if None, all bands are returned

        """

        assert self.has_open_scene, "No scene is open"
        assert self.__bands is not None, "No bands are loaded, this should not happen"

        if bands is None:
            return self.__bands

        band_idx = [band.value[1] for band in bands]
        return self.__bands[band_idx]

    @accumulate_time
    def get_masks(self, mask_type: Masks = Masks.AUTO_GENERATED) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the masks and its coverage for the current scene.

        Masks can be lazy loaded, so they are not loaded until this function is called.
        @arg mask: the mask to return, if not set the auto generated mask is returned
        """

        if self.__masks is not None and mask_type in self.__masks:
            return self.__masks[mask_type]

        mask = self.__mask_handler.load_data(
            tile_id=self.tile_id,
            date=self.date,
            resolution=self.resolution,
            mask_type=mask_type,
        )

        if self.__masks is None:
            self.__masks = {}

        self.__masks[mask_type] = mask
        return mask

    @accumulate_time
    def get_auxiliary_data(self, auxiliary_data: AuxiliaryData):
        """

        Returns the auxiliary data for the current scene.

        """

        assert self.resolution == 10, "Auxiliary data is only available for 10m resolution"

        if self.__auxiliary_data is not None and auxiliary_data in self.__auxiliary_data:
            return self.__auxiliary_data[auxiliary_data]

        if self.__auxiliary_data is None:
            self.__auxiliary_data = {}

        print(" » Loading auxiliary data: {}".format(auxiliary_data))
        data = self.__auxiliary_reader.load_data(tile_id=self.tile_id, auxiliary_data=auxiliary_data)
        data = self._normalize_auxiliary_data(data, auxiliary_data)
        self.__auxiliary_data[auxiliary_data] = data

        assert data.shape == self.__shape, \
            f"Auxiliary data has wrong shape: {data.shape} but the expected shape is {self.__shape}"
        return data

    @accumulate_time
    def save_masks(self, mask_data, mask_type: Masks = Masks.AUTO_GENERATED):
        """
        Saves the masks in the memory.

        @arg mask_data: the mask data to save
        @arg mask: the mask type

        """

        assert len(mask_data.shape) == 3, "Mask data must be 3 dimensional"
        assert mask_data.shape[0] == 2, "Mask data must have 2 channels: mask and coverage"

        self.__mask_handler.save_data(
            data=mask_data,
            mask_type=mask_type,
            tile_id=self.tile_id,
            date=self.date,
            resolution=self.resolution,
            profile=self.__profile
        )

    @accumulate_time
    def get_exo_labs(self, my_encoding=False):
        assert self.resolution == 10, "Auxiliary data is only available for 10m resolution"

        if self.__exo_labs is None:
            self.__exo_labs = self.__exo_labs_reader.load_data(tile_id=self.tile_id, date=self.date)

        assert self.__exo_labs.shape == self.__shape, \
            f"Auxiliary data has wrong shape: {self.__exo_labs.shape} but the expected shape is {self.__shape}"

        data = self.__exo_labs

        if my_encoding:
            encoded_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
            encoded_data[data == 3] = 2  # clouds
            encoded_data[data == 4] = 1  # snow
            encoded_data[data == 6] = 3  # water

            return encoded_data

        return data

    @accumulate_time
    def get_satellite_data_coverage(self):

        assert self.__satellite_data_coverage is not None, "Satellite data coverage is not loaded"
        return self.__satellite_data_coverage

    @accumulate_time
    def __get_satellite_data_coverage(self, bands: np.ndarray):
        """

        Returns the coverage of the loaded data

        """

        # open and pre-process all bands (except additional metadata, i.g. elevation)
        data_coverage = np.zeros(bands[0].shape, dtype=np.uint8)

        # we keep only regions where some band has data
        for band in bands:
            data_coverage |= (band > 0)

        # add a safety margin of 256 pixels around every data_coverage == 0 pixel
        data_coverage = ~data_coverage
        data_coverage = cv2.dilate(data_coverage, np.ones((128, 128), np.uint8), iterations=1)
        data_coverage = ~data_coverage

        data_coverage[0:64, :] = 0
        data_coverage[-64:, :] = 0
        data_coverage[:, 0:64] = 0
        data_coverage[:, -64:] = 0

        return data_coverage

    @accumulate_time
    def close_scene(self):
        """
        Closes the current scene and clears the memory.
        """

        print(" » Closing scene for tile_id: {} and date: {}".format(self.tile_id, self.date))

        self.has_open_scene = False

        self.__bands = None
        self.__masks = None
        self.__auxiliary_data = None
        self.__exo_labs = None

        self.__shape = None
        self.__profile = None

    def report_time(self):
        print(f"\nTiming report for {self.date}:")
        print(self.cct.__str__())
        print()
