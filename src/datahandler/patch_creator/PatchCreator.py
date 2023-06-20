import abc

import numpy as np
from pytictac import ClassTimer, accumulate_time, ClassContextTimer

from src.datahandler.DataHandler import DataHandler
from src.datahandler.auxiliary_reader.AuxiliaryReader import AuxiliaryData
from src.datahandler.satallite_reader.SentinelL1CReader import Bands


class PatchCreator(abc.ABC):

    def __init__(self,
                 dataloader: DataHandler,
                 patch_size: int = 256,
                 bands: list[Bands] = None,
                 auxiliary_data: list[AuxiliaryData] = None,
                 timer_enabled=True):

        self.dataloader = dataloader
        self.patch_size = patch_size
        self.bands = bands
        self.auxiliary_data = auxiliary_data if auxiliary_data is not None else []

        self.timer_enabled = timer_enabled
        self.cct = ClassTimer(objects=[self], names=["PatchCreator"], enabled=timer_enabled)

    @accumulate_time
    def _switch_scene(self, tile_id: str, date: str) -> None:

        """
        Switches the scene to the given tile and date.
        """

        if self.dataloader.tile_id != tile_id or self.dataloader.date != date:
            if self.dataloader.has_open_scene:
                self.dataloader.close_scene()
            self.dataloader.open_scene(tile_id, date)

    @accumulate_time
    def get_patch(self, tile_id: str, date: str, coords: tuple[int, int], resolution: int = 10, include_mask=True) -> \
            tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Returns a new patch from the given tile and date.
        """

        assert resolution == 10, "Only 10m resolution is supported for patches"
        self._switch_scene(tile_id, date)

        with ClassContextTimer(parent_obj=self, block_name="get_bands", parent_method_name="get_patch"):
            bands = self.dataloader.get_bands(bands=self.bands)

        mask_patch = None
        if include_mask:
            mask, _ = self.dataloader.get_masks()
            mask_patch = mask[coords[0]:coords[0] + self.patch_size, coords[1]:coords[1] + self.patch_size]

        with ClassContextTimer(parent_obj=self, block_name="get_auxiliary_data", parent_method_name="get_patch"):
            assert len(self.auxiliary_data) == 1, "Only one auxiliary data is supported for patches"
            auxiliary_data = self.dataloader.get_auxiliary_data(auxiliary_data=self.auxiliary_data[0])
            auxiliary_data = np.expand_dims(auxiliary_data, axis=0)

        with ClassContextTimer(parent_obj=self, block_name="crop_patches", parent_method_name="get_patch"):
            bands_patch = bands[:, coords[0]:coords[0] + self.patch_size, coords[1]:coords[1] + self.patch_size]
            auxiliary_data_patch = auxiliary_data[:, coords[0]:coords[0] + self.patch_size,
                                   coords[1]:coords[1] + self.patch_size]

        # combine bands with auxiliary data
        if len(auxiliary_data) > 0:
            data_patch = np.concatenate((bands_patch, auxiliary_data_patch), axis=0)
        else:
            data_patch = bands_patch

        patch = (data_patch, mask_patch) if include_mask else data_patch
        return patch

    def report_timing(self):
        assert self.timer_enabled is True, "Timer is not enabled"
        print(self.dataloader.report_time())
        print(self.cct.__str__())
        print()

    def plot_patch_centers(self, tile_id: str, date: str):
        pass
