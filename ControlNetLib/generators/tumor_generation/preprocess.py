import cv2
import numpy as np
import sys
import torch
import einops

sys.path.insert(1, 'ControlNet')
from annotator.canny import CannyDetector
from annotator.util import resize_image, HWC3

image_resolution = 512

def preprocess(image: np.ndarray, tumor_location: tuple, tumor_size: int) -> torch.tensor:
    """
    Preprocesses an image by converting it to grayscale, resizing, applying Canny edge detection, and adding a synthetic tumor.

    Args:
        image (np.ndarray): The input image, expected to be in BGR format.
        tumor_location (tuple[int, int]): The (x, y) coordinates for the center of the synthetic tumor.
        tumor_size (int): The radius of the synthetic tumor to be added.

    Returns:
        torch.tensor: The processed image with edges detected and a synthetic tumor added. As tensor.
    """
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (image_resolution, image_resolution))

    apply_canny = CannyDetector()
    edges = apply_canny(image, 10, 20)
    
    # Add synthetic tumor
    cv2.circle(edges, (tumor_location[0], tumor_location[1]), tumor_size, 0, -1)
    cv2.circle(edges, (tumor_location[0], tumor_location[1]), tumor_size, 255, 1)

    img = resize_image(HWC3(edges), image_resolution)
    
    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    control = einops.rearrange(control, 'h w c -> c h w')
    control = torch.reshape(control, (1, 3, image_resolution, image_resolution))
    
    return control