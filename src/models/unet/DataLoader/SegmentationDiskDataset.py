# all PyTorch datasets must inherit from this base dataset class.
import cv2
import numpy as np
from torchvision.transforms import Compose

from src.models.unet.configs.config import NUM_ENCODED_CHANNELS, NUM_CLASSES
from src.models.unet.DataLoader.SegmentationDataset import SegmentationDataset
from src.models.unet.augmentation.Augmentation import Augmentation


class SegmentationDiskDataset(SegmentationDataset):
    """

    This class is used to load the data from the disk.
    If you want to create the patches on the fly from the original images,
    you should use the SegmentationDataset.py class.

    """

    def __init__(self,
                 image_paths: list[str],
                 mask_paths: list[str],
                 transforms: Compose,
                 apply_augmentations: bool = True,
                 augmentations: list[Augmentation] = None):
        super().__init__(
            transforms=transforms,
            apply_augmentations=apply_augmentations,
            augmentations=augmentations
        )

        self.imagePaths = image_paths
        self.maskPaths = mask_paths

        assert len(self.imagePaths) == len(self.maskPaths), "Number of images and masks must match."
        print(f"Dataloader initialized with {len(self.imagePaths)} patches.")

    def __len__(self) -> int:
        # return the number of total samples contained in the dataset
        return len(self.maskPaths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """

        :param idx: index of the image to be loaded
        :return: the corresponding sample from the dataset
        """

        # grab the image path from the current index
        image_path, mask_path = self.imagePaths[idx], self.maskPaths[idx]

        image = np.load(image_path)
        image = np.moveaxis(image, 0, -1)  # TODO: can we move this to the dataset creation step?

        mask = np.load(mask_path)

        mask = mask.astype(int)
        mask = mask * NUM_ENCODED_CHANNELS // 255

        encoded_mask = [None] * NUM_CLASSES
        for cid in range(NUM_CLASSES):
            encoded_mask[cid] = cv2.inRange(mask, cid, cid)  # add + 1 behind cid to exclude background

        return self._transform_and_augment(image, encoded_mask)
