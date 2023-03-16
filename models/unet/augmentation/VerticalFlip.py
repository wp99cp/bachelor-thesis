import torch

from augmentation.Augmentation import Augmentation


class VerticalFlip(Augmentation):

    def __init__(self, prob=0.5):
        super().__init__(prob)

    def apply(self, image, mask):
        if self._rand_choice():
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        return image, mask
