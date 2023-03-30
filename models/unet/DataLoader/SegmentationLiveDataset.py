import os
import sys

import cv2
import numpy as np
from torchvision.transforms import Compose

from DataLoader.SegmentationDataset import SegmentationDataset
from augmentation.Augmentation import Augmentation
from configs.config import NUM_CLASSES

# import the necessary packages form the pre-processing/image_splitter
sys.path.insert(0, os.environ['BASE_DIR'] + '/pre-processing/image_splitter')
# noinspection PyUnresolvedReferences
from config import SAMPLES_PER_DATE
from RandomPatchCreator import RandomPatchCreator


class SegmentationLiveDataset(SegmentationDataset):
    """

    This class creates the patches on the fly from the original images.
    If you want to load the data from the disk, i.g. the pre-computed patches,
    you should use the SegmentationDiskDataset class.

    """

    def __init__(self,
                 dates: list[str],
                 transforms: Compose,
                 apply_augmentations: bool = True,
                 augmentations: list[Augmentation] = None):
        """
        :param dates: The dates from which the patches should be created.
        :param transforms: transformations to be applied to the patches
        :param apply_augmentations: if True, the augmentations will be applied to the patches
        :param augmentations: augmentations to be applied to the patches
        """

        super().__init__(
            transforms=transforms,
            apply_augmentations=apply_augmentations,
            augmentations=augmentations
        )

        self.patch_creator = RandomPatchCreator(dates=dates)

        print(f"Dataloader initialized with the following dates: {dates}")

        # TODO: Calculate free memory...
        self.patch_creator.open_date(dates[0])
        self.patch_creator.open_date(dates[1])

    def __len__(self) -> int:
        """
        As the patches are created on the fly, the length of the dataset is not known.
        However, this function is needed by the PyTorch dataloader. We therefore return
        a preset constant value.

        :return: SAMPLES_PER_DATE * len(self.dates)
        """

        return SAMPLES_PER_DATE * self.patch_creator.get_number_of_dates()

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        :param idx: index of the image to be loaded
        :return: the corresponding sample from the dataset
        """

        patch_img, mask = self.patch_creator.next()

        # one hot encode the mask
        encoded_mask = [None] * NUM_CLASSES
        for cid in range(NUM_CLASSES):
            encoded_mask[cid] = cv2.inRange(mask, cid, cid)  # add + 1 behind cid to exclude background

        return self._transform_and_augment(patch_img, encoded_mask)
