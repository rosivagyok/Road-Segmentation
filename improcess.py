import numpy as np
import random
import cv2
from os.path import join, expanduser
import matplotlib.pyplot as plt

def perform_augmentation(batch_x, batch_y):
    
    def mirror(x):
        return x[:, ::-1, :]

    # apply random changes to Hue, Saturation and Brightness. This can be done in HSV color space, therefore color space is converted from RGB
    def augment_in_hsv_space(x_hsv):
        x_hsv = np.float32(cv2.cvtColor(x_hsv, cv2.COLOR_RGB2HSV))
        x_hsv[:, :, 0] = x_hsv[:, :, 0] * random.uniform(0.9, 1.1)   # change hue
        x_hsv[:, :, 1] = x_hsv[:, :, 1] * random.uniform(0.5, 2.0)   # change saturation
        x_hsv[:, :, 2] = x_hsv[:, :, 2] * random.uniform(0.5, 2.0)   # change brightness
        x_hsv = np.uint8(np.clip(x_hsv, 0, 255))

        # Convert back to RGB color space
        return cv2.cvtColor(x_hsv, cv2.COLOR_HSV2RGB)

    batch_x_aug = np.copy(batch_x)
    batch_y_aug = np.copy(batch_y)

    for b in range(batch_x_aug.shape[0]):

        # Random mirroring, based on coinflip choice
        should_mirror = random.choice([True, False])
        if should_mirror:
            batch_x_aug[b] = mirror(batch_x[b])
            batch_y_aug[b] = mirror(batch_y[b])

        # Random change in image values (hue, saturation, brightness)
        batch_x_aug[b] = augment_in_hsv_space(batch_x_aug[b])

    return batch_x_aug, batch_y_aug
