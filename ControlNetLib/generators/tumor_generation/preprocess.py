import cv2
import numpy as np
import sys

sys.path.insert(1, 'ControlNet')
from annotator.canny import CannyDetector


def preprocess(image: np.ndarray, tumor_location: tuple[int, int], tumor_size: int) -> np.ndarray:
    """
    Preprocesses an image by converting it to grayscale, resizing, applying Canny edge detection, and adding a synthetic tumor.

    Args:
        image (np.ndarray): The input image, expected to be in BGR format.
        tumor_location (tuple[int, int]): The (x, y) coordinates for the center of the synthetic tumor.
        tumor_size (int): The radius of the synthetic tumor to be added.

    Returns:
        np.ndarray: The processed image with edges detected and a synthetic tumor added.
    """
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))

    apply_canny = CannyDetector()
    edges = apply_canny(image, 10, 20)
    
    # Add synthetic tumor
    cv2.circle(edges, (tumor_location[0], tumor_location[1]), tumor_size, 0, -1)
    cv2.circle(edges, (tumor_location[0], tumor_location[1]), tumor_size, 255, 1)

    return edges