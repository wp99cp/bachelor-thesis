# all PyTorch datasets must inherit from this base dataset class.
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import NUM_ENCODED_CHANNELS, NUM_CLASSES, CHANNEL_DROPOUT_PROB, IMAGE_FLIP_PROB, \
    PATCH_COVERING_PROB, COVERED_PATCH_SIZE_MIN, COVERED_PATCH_SIZE_MAX


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
            masks[cid] = cv2.inRange(mask, cid + 1, cid + 1)

        masks = np.stack(masks, axis=-1)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            masks = self.transforms(masks)

        if self.apply_augmentations:

            # apply the augmentation transforms
            flip_prob = np.random.rand()
            if flip_prob < IMAGE_FLIP_PROB:
                image = torch.flip(image, dims=[1])
                masks = torch.flip(masks, dims=[1])

            flip_prob = np.random.rand()
            if flip_prob < IMAGE_FLIP_PROB:
                image = torch.flip(image, dims=[2])
                masks = torch.flip(masks, dims=[2])

            # Channel Dropout selects a random channel and sets it to zero
            # This happens with a probability of 0.3
            for channel_idx in range(image.shape[0]):
                channel_dropout_prob = np.random.rand()
                if channel_dropout_prob < CHANNEL_DROPOUT_PROB:
                    image[channel_idx, :, :] = 0

            # cover a random patch of the image (i.g. setting all channels and the mask to zero)
            channel_cover_prob = np.random.rand()
            if channel_cover_prob < PATCH_COVERING_PROB:
                patch_size = np.random.randint(COVERED_PATCH_SIZE_MIN, COVERED_PATCH_SIZE_MAX)

                # select a random patch
                x = np.random.randint(0, image.shape[1] - patch_size)
                y = np.random.randint(0, image.shape[2] - patch_size)

                # set the patch to zero
                image[:, x:x + patch_size, y:y + patch_size] = 0
                masks[:, x:x + patch_size, y:y + patch_size] = 0

        # return a tuple of the image and its mask
        return image, masks
