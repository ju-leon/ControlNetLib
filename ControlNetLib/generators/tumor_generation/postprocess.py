import cv2
import numpy as np
def postprocess(image: np.ndarray) -> np.ndarray:    
    """
    Postprocessing function for the images generated by the Tumor Generator.

    Args:
        image (ndarray): Input image, should be a 3-channel image in BGR format.
    Returns:
        ndarray: A 1-channel grayscale image with the same height and width as the input image.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)

    return image