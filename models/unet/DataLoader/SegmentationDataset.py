# all PyTorch datasets must inherit from this base dataset class.
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import NUM_ENCODED_CHANNELS, NUM_CLASSES


class SegmentationDataset(Dataset):

    def __init__(self, imagePaths, maskPaths, transforms, apply_augmentations=True):
        # store the image and mask filepaths, and augmentation transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths

        # this is a set of transformations that will be applied to the image and mask
        self.transforms = transforms
        self.apply_augmentations = apply_augmentations

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.maskPaths)

    def __getitem__(self, idx):
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
            masks[cid] = cv2.inRange(mask, cid, cid)

        masks = np.stack(masks, axis=-1)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            masks = self.transforms(masks)

        if self.apply_augmentations:

            # apply the augmentation transforms
            flip_prob = np.random.rand()
            if flip_prob > 0.5:
                image = torch.flip(image, dims=[1])
                masks = torch.flip(masks, dims=[1])

            flip_prob = np.random.rand()
            if flip_prob > 0.5:
                image = torch.flip(image, dims=[2])
                masks = torch.flip(masks, dims=[2])

        # return a tuple of the image and its mask
        return image, masks
