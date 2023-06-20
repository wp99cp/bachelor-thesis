from numpy import ndarray
from pytictac import accumulate_time

from src.datahandler.DataHandler import DataHandler
from src.datahandler.auxiliary_reader.AuxiliaryReader import AuxiliaryData
from src.datahandler.patch_creator.PatchCreator import PatchCreator
from src.datahandler.satallite_reader.SentinelL1CReader import Bands

MAX_TRIES = 1_000


class CoveragePatchCreator(PatchCreator):

    def __init__(self,
                 dataloader: DataHandler,
                 patch_size: int,
                 auxiliary_data: list[AuxiliaryData],
                 bands: list[Bands] = None,
                 timer_enabled=True
                 ):
        super().__init__(
            dataloader,
            patch_size=patch_size,
            bands=bands,
            auxiliary_data=auxiliary_data,
            timer_enabled=timer_enabled
        )

        self.old_coords = (0, 0)
        self.date = None
        self.tile_id = None

        self.shape = None

        self.half_patch_size = self.patch_size // 2

    @accumulate_time
    def get_next_patch(self, tile_id: str, date: str, resolution: int = 10, include_mask=True) -> \
            tuple[tuple[int, int], tuple[ndarray, ndarray]] | tuple[tuple[int, int], ndarray]:

        # change the scene if necessary
        # this may be slow: make sure to not overlap calls to this function
        # requesting patches from different tiles or dates
        self._switch_scene(tile_id, date)

        if (self.tile_id != tile_id) or (self.date != date) or (self.shape is None):
            self.tile_id = tile_id
            self.date = date
            self.shape = self.dataloader.get_bands()[0].shape

        # get random coordinates
        coords = self.old_coords

        coords = (coords[0] + self.half_patch_size, coords[1])
        if coords[0] >= self.shape[0]:
            coords = (0, coords[1] + self.half_patch_size)
            if coords[1] >= self.shape[1]:
                raise Exception("No more patches available")

        self.old_coords = coords

        patch = super().get_patch(tile_id, date, coords, resolution, include_mask)
        return coords, patch

    @accumulate_time
    def get_patch_count(self):
        """
        Returns the number of patches needed to cover the whole scene.
        """

        x_count = self.dataloader.get_bands()[0].shape[0] // self.half_patch_size
        y_count = self.dataloader.get_bands()[0].shape[1] // self.half_patch_size
        return x_count * y_count
