from multiprocessing import Lock

import numpy as np

from config import NUM_ENCODED_CHANNELS, IMAGE_SIZE


class PixelClassCounter:

    def __init__(self,
                 num_encoded_channels: int):
        self.__num_encoded_channels = num_encoded_channels

        self._class_distribution = {}
        self.lock = Lock()

    def __register_date(self, date: str):

        with self.lock:
            self._class_distribution[date] = {
                'mask_count': 0,
                'distribution_sum': [0.0] * self.__num_encoded_channels
            }

        assert date in self._class_distribution.keys()

    def update(self, mask: np.ndarray, date: str):

        # Count the number of pixels in each class
        pixel_count = [0.0] * NUM_ENCODED_CHANNELS
        for j in range(NUM_ENCODED_CHANNELS):
            pixel_count[j] += np.sum(mask == j * (255 / NUM_ENCODED_CHANNELS))

        normalized_pixel_count = np.array(pixel_count) / (IMAGE_SIZE ** 2)

        if date not in self._class_distribution.keys():
            self.__register_date(date)

        # store the distribution thread safe
        with self.lock:
            self._class_distribution[date]['mask_count'] += 1
            self._class_distribution[date]['distribution_sum'] += normalized_pixel_count

    def get_class_distribution(self, date: str = None):

        if date is None:

            # return weighted average
            distribution_sum = np.array([0.0] * self.__num_encoded_channels)
            weight_sum = 0

            for date in self._class_distribution.keys():
                weight = self._class_distribution[date]['mask_count']
                weight_sum += weight
                distribution_sum += self._class_distribution[date]['distribution_sum']

            return distribution_sum / weight_sum

        assert date in self._class_distribution.keys(), f"Date {date} not found in {self._class_distribution.keys()}"
        return self._class_distribution[date]['distribution_sum'] / self._class_distribution[date]['mask_count']
