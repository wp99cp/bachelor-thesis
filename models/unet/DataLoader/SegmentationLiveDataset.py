import os
import random
import sys
from queue import Queue

import cv2
import numpy as np
from torchvision.transforms import Compose

from DataLoader.SegmentationDataset import SegmentationDataset
from augmentation.Augmentation import Augmentation
from configs.config import NUM_CLASSES, BATCH_PREFETCHING, BATCH_MIXTURE, BATCH_SIZE

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
                 augmentations: list[Augmentation] = None,
                 mixture: int = BATCH_MIXTURE,
                 mixture_queue_size: int = int(BATCH_SIZE / BATCH_MIXTURE) * BATCH_PREFETCHING
                 ):
        """
        :param dates: The dates from which the patches should be created.
        :param transforms: transformations to be applied to the patches
        :param apply_augmentations: if True, the augmentations will be applied to the patches
        :param augmentations: augmentations to be applied to the patches
        :param mixture: number of different dates used concurrently for fetching patches
        """

        super().__init__(
            transforms=transforms,
            apply_augmentations=apply_augmentations,
            augmentations=augmentations
        )

        self.__dates = dates
        self.patch_creator = RandomPatchCreator(dates=dates)

        print(f"\nDataloader initialized with the following dates: {dates}")
        print(f"- number mixture queues: {mixture}")
        print(f"- mixture queue size: {mixture_queue_size}\n")

        self.__mixture = min(mixture, len(dates))
        self.__mixture_queue_size = mixture_queue_size

        self.__queues = []
        for i in range(self.__mixture):
            self.__queues.append(Queue(maxsize=mixture_queue_size))

        # prefill the queues
        for i in range(self.__mixture):
            self.__fill_queue(i, dates[i])

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

        # select a random queue
        queue_idx = np.random.randint(0, self.__mixture)

        # pop the next patch from the queue
        patch_img, mask = self.__queues[queue_idx].get()

        # fill the queue if it is empty with a new random date
        if self.__queues[queue_idx].empty():
            self.__fill_queue(queue_idx, random.choice(self.__dates))

        # one hot encode the mask
        encoded_mask = [None] * NUM_CLASSES
        for cid in range(NUM_CLASSES):
            encoded_mask[cid] = cv2.inRange(mask, cid, cid)  # add + 1 behind cid to exclude background

        return self._transform_and_augment(patch_img, encoded_mask)

    def __fill_queue(self, queue_idx: int, date: str):

        print(f"\nfilling queue {queue_idx} with patches from date {date}")
        self.patch_creator.open_date(date)

        for i in range(self.__mixture_queue_size):
            patch = self.patch_creator.random_patch(date)

            assert self.__queues[queue_idx].full() is False, f"Queue {queue_idx} is full"
            self.__queues[queue_idx].put(patch)

        print(f"Finished filling queue {queue_idx} with patches from date {date}\n")
