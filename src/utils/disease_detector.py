# src/utils/disease_detector.py

import cv2
import numpy as np
from PIL import Image


def detect_disease_spots(original_image, sensitivity=50):
    """
    Detect and outline disease spots on a grape leaf image using color thresholding.

    Disease spots are identified by their characteristic colors:
    - Dark brown to black spots indicate Black rot
    - Brown and rust colored patches indicate Esca or Leaf blight

    Parameters:
        original_image: PIL image of the grape leaf
        sensitivity:    upper brightness threshold for dark spot detection.
                        lower value = only very dark spots detected.
                        higher value = more spots detected including lighter ones.

    Returns:
        result:      numpy array (224, 224, 3) with disease spots outlined in red
        spot_count:  number of disease spots found
        mask:        binary mask showing detected disease regions
    """

    # Resize and convert to BGR for OpenCV processing
    img     = np.array(original_image.resize((224, 224)))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to HSV color space for more reliable color segmentation
    # HSV separates color (hue) from brightness (value) making thresholding more robust
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Detect dark brown and black spots (Black rot)
    # Low value channel = dark pixels regardless of hue
    lower_dark = np.array([0,   0,   0])
    upper_dark = np.array([180, 255, sensitivity])

    # Detect brown and rust colored patches (Esca, Leaf blight)
    # Hue range 5-25 covers brown and orange-brown tones
    lower_brown = np.array([5,  50,  50])
    upper_brown = np.array([25, 255, 200])

    # Combine both masks to cover all disease spot types
    mask_dark  = cv2.inRange(hsv, lower_dark,  upper_dark)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask       = cv2.bitwise_or(mask_dark, mask_brown)

    # Remove noise using morphological operations
    # MORPH_OPEN  removes small isolated noise pixels
    # MORPH_CLOSE fills small holes inside detected spots
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find external contours around each disease region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result     = img.copy()
    spot_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip very small regions that are likely noise not disease spots
        if area > 10:
            spot_count += 1

            # Draw red outline around the disease spot
            cv2.drawContours(result, [contour], -1, (255, 0, 0), 2)

            # Draw semi-transparent red fill inside the contour
            overlay = result.copy()
            cv2.drawContours(overlay, [contour], -1, (255, 50, 50), -1)
            result  = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)

    return result, spot_count, mask