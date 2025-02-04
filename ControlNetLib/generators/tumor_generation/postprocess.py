import cv2

def postprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize histogram
    image = cv2.equalizeHist(image)

    return image