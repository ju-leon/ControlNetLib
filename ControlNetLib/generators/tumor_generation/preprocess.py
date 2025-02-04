import cv2
import numpy as np

def preprocess(image, tumor_location, tumor_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image