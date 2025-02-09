import logging
import cv2
import numpy as np

# Setup logger for this module
logger = logging.getLogger(__name__)

def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from the given file path.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        np.ndarray: The image read from disk.
        
    Raises:
        ValueError: If the image cannot be read.
    """
    logger.info("Reading image from path: %s", image_path)
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Failed to read image from %s", image_path)
        raise ValueError(f"Image at {image_path} could not be read.")
    logger.debug("Image shape: %s", image.shape)
    return image

def resize_image(image: np.ndarray, width: int = None, height: int = None, interpolation=cv2.INTER_AREA) -> np.ndarray:
    """
    Resize an image to the specified width and/or height.
    
    Args:
        image (np.ndarray): Input image.
        width (int): Desired width.
        height (int): Desired height.
        interpolation: Interpolation method for resizing.
    
    Returns:
        np.ndarray: The resized image.
    """
    logger.info("Resizing image.")
    if width is None and height is None:
        logger.debug("No dimensions provided for resizing; returning original image.")
        return image

    (h, w) = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)
    
    resized = cv2.resize(image, dim, interpolation=interpolation)
    logger.debug("Resized image shape: %s", resized.shape)
    return resized

def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an image from BGR (OpenCV default) to RGB.
    
    Args:
        image (np.ndarray): Input image in BGR format.
    
    Returns:
        np.ndarray: Image in RGB format.
    """
    logger.info("Converting image from BGR to RGB.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the pixel values of an image to the [0, 1] range.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Normalized image.
    """
    logger.info("Normalizing image pixel values to [0, 1].")
    normalized = image.astype('float32') / 255.0
    return normalized