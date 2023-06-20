import numpy as np

from src.datahandler.DataHandler import DataHandler
from src.datahandler.auxiliary_reader.AuxiliaryReader import AuxiliaryData
from src.datahandler.patch_creator.PatchCreator import PatchCreator
from src.datahandler.satallite_reader.SentinelL1CReader import Bands

MAX_TRIES = 1_000


class RandomPatchCreator(PatchCreator):

    def __init__(self,
                 dataloader: DataHandler,
                 patch_size: int,
                 bands: list[Bands],
                 auxiliary_data: list[AuxiliaryData]):
        super().__init__(
            dataloader,
            patch_size=patch_size,
            bands=bands,
            auxiliary_data=auxiliary_data
        )

    def __get_random_coordinates(self) -> tuple[int, int]:

        _, mask_coverage = self.dataloader.get_masks()

        count = 0
        while count < MAX_TRIES:
            count += 1

            # sample a random patch
            x = np.random.randint(0, mask_coverage.shape[0] - self.patch_size)
            y = np.random.randint(0, mask_coverage.shape[1] - self.patch_size)

            # check if the patch is valid
            if np.sum(mask_coverage[x:x + self.patch_size, y:y + self.patch_size]) == \
                    self.patch_size ** 2:
                return int(x), int(y)

        valid_pixel_count = np.sum(mask_coverage)
        total_pixel_count = mask_coverage.shape[0] * mask_coverage.shape[1]

        print(f"Could not find a valid patch after {MAX_TRIES} tries. "
              f"Valid pixel count: {valid_pixel_count} / {total_pixel_count}")
        raise Exception("Could not find a valid patch")

    def get_random_patch(self, tile_id: str, date: str, resolution: int = 10, include_mask=True) -> tuple[
        np.ndarray, np.ndarray]:

        # change the scene if necessary
        # this may be slow: make sure to not overlap calls to this function
        # requesting patches from different tiles or dates
        self._switch_scene(tile_id, date)

        # get random coordinates
        coords = self.__get_random_coordinates()
        patch = super().get_patch(tile_id, date, coords, resolution, include_mask)
        return patch
