import numpy as np

from src.models.unet.configs.config import THRESHOLD


def get_encoded_prediction(predicted_mask, thresholded_prediction=False):
    print(f"Min/Max of predicted_mask: {np.min(predicted_mask)}/{np.max(predicted_mask)}")
    if thresholded_prediction:
        predicted_mask[:, :, :] = (predicted_mask[:, :, :] > THRESHOLD).astype(int)
    return np.argmax(predicted_mask, axis=0)
