import cv2
import numpy as np
import sys

sys.path.insert(1, 'ControlNet')
from annotator.canny import CannyDetector


def preprocess(image, tumor_location, tumor_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))

    apply_canny = CannyDetector()
    edges = apply_canny(image, 10, 20)
    
    # Add synthetic tumor
    cv2.circle(edges, (tumor_location[0], tumor_location[1]), tumor_size, 0, -1)
    cv2.circle(edges, (tumor_location[0], tumor_location[1]), tumor_size, 255, 1)

    return edges