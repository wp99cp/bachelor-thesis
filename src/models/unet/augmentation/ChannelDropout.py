from augmentation.Augmentation import Augmentation


class ChannelDropout(Augmentation):

    def __init__(self, prob=0.3):
        super().__init__(prob)

    def apply(self, image, mask):

        # Channel Dropout selects a random channel and sets it to zero
        # This happens with a probability of 0.3
        for channel_idx in range(image.shape[0]):
            if self._rand_choice():
                image[channel_idx, :, :] = 0

        return image, mask
