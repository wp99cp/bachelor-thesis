from abc import ABC, abstractmethod

import numpy as np


class Augmentation(ABC):

    def __init__(self, prob=0.5):
        self.prob = prob

    def _rand_choice(self):
        return self.prob >= np.random.rand()

    @abstractmethod
    def apply(self, image, mask):
        return image, mask
