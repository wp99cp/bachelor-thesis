# all PyTorch datasets must inherit from this base dataset class.
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from augmentation.Augmentation import Augmentation
from configs.config import NUM_ENCODED_CHANNELS, NUM_CLASSES


class SegmentationDataset(Dataset):

    def __init__(self,
                 image_paths: list[str],
                 mask_paths: list[str],
                 transforms: Compose,
                 apply_augmentations: bool = True,
                 augmentations: list[Augmentation] = None):

        if augmentations is None:
            augmentations = []

        self.imagePaths = image_paths
        self.maskPaths = mask_paths

        # this is a set of transformations that will be applied to the image and mask
        self.transforms = transforms
        self.apply_augmentations = apply_augmentations
        self.augmentations = augmentations

    def __len__(self) -> int:
        # return the number of total samples contained in the dataset
        return len(self.maskPaths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """

        :param idx: index of the image to be loaded
        :return: the corresponding sample from the dataset
        """

        # grab the image path from the current index
        image_path = self.imagePaths[idx]

        image = np.load(image_path)
        image = image['arr_0'].astype(np.float32)
        image = image / 255.0
        image = np.moveaxis(image, 0, -1)

        # load the corresponding ground-truth segmentation mask in grayscale mode
        mask = cv2.imread(self.maskPaths[idx], cv2.IMREAD_UNCHANGED)

        mask = mask.astype(int)
        mask = mask * NUM_ENCODED_CHANNELS // 255

        masks = [None] * NUM_CLASSES
        for cid in range(NUM_CLASSES):
            masks[cid] = cv2.inRange(mask, cid + 1, cid + 1)

        masks = np.stack(masks, axis=-1)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            masks = self.transforms(masks)

        if self.apply_augmentations:

            for aug in self.augmentations:
                image, masks = aug.apply(image, masks)

        # return a tuple of the image and its mask
        return image, masks
