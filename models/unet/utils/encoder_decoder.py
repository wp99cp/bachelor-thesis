import numpy as np

from configs.config import THRESHOLD, THRESHOLDED_PREDICTION


def get_encoded_prediction(predicted_mask):
    print(f"Min/Max of predicted_mask: {np.min(predicted_mask)}/{np.max(predicted_mask)}")
    if THRESHOLDED_PREDICTION:
        predicted_mask[:, :, :] = (predicted_mask[:, :, :] > THRESHOLD).astype(int)
    return np.argmax(predicted_mask, axis=0)
