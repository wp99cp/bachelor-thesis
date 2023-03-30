from abc import ABC

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from augmentation.Augmentation import Augmentation


class SegmentationDataset(Dataset, ABC):

    def __init__(self,
                 transforms: Compose,
                 apply_augmentations: bool = True,
                 augmentations: list[Augmentation] = None):
        if augmentations is None:
            augmentations = []

        # this is a set of transformations that will be applied to the image and mask
        self.transforms = transforms
        self.apply_augmentations = apply_augmentations
        self.augmentations = augmentations

    def _transform_and_augment(self, patch_img: np.ndarray, encoded_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        encoded_mask = np.stack(encoded_mask, axis=-1)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            patch_img = self.transforms(patch_img)
            encoded_mask = self.transforms(encoded_mask)

        if self.apply_augmentations:

            for aug in self.augmentations:
                patch_img, encoded_mask = aug.apply(patch_img, encoded_mask)

        # return a tuple of the image and its mask
        return patch_img, encoded_mask
