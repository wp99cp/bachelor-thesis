import numpy as np

from augmentation.Augmentation import Augmentation


class RandomErasing(Augmentation):

    def __init__(self, min_size, max_size, prob=0.3):
        super().__init__(prob)

        self.min_size = min_size
        self.max_size = max_size

    def apply(self, image, mask):
        if self._rand_choice():
            patch_size = np.random.randint(self.min_size, self.max_size)

            # select a random patch
            x = np.random.randint(0, image.shape[1] - patch_size)
            y = np.random.randint(0, image.shape[2] - patch_size)

            # set the patch to zero
            image[:, x:x + patch_size, y:y + patch_size] = 0
            mask[:, x:x + patch_size, y:y + patch_size] = 0

        return image, mask
