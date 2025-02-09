import os
import glob
import cv2
import numpy as np
import logging

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

def draw_detections(image: np.ndarray, boxes: np.ndarray, landmarks: list) -> np.ndarray:
    """
    Draw bounding boxes and landmarks on the image.
    
    Args:
        image (np.ndarray): Original image.
        boxes (np.ndarray): Array of bounding boxes.
        landmarks (list): List of landmarks for each detected face.
    
    Returns:
        np.ndarray: Annotated image.
    """
    annotated = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if landmarks is not None and i < len(landmarks):
            for (x, y) in landmarks[i]:
                cv2.circle(annotated, (int(x), int(y)), 2, (0, 0, 255), -1)
    return annotated

def create_composite_image(annotated_img: np.ndarray, face_rows: list, target_width: int = 448) -> np.ndarray:
    """
    Combine the annotated image and the per-face rows into a single composite image.
    
    Args:
        annotated_img (np.ndarray): Annotated original image.
        face_rows (list): List of images (rows) each showing the cropped and aligned face.
        target_width (int): The width to which face rows are set.
    
    Returns:
        np.ndarray: The final composite image.
    """
    annotated_h, annotated_w = annotated_img.shape[:2]
    new_height = int(annotated_h * (target_width / annotated_w))
    annotated_resized = cv2.resize(annotated_img, (target_width, new_height))
    
    if face_rows:
        faces_composite = cv2.vconcat(face_rows)
        margin = 10
        margin_img = np.ones((margin, target_width, 3), dtype=np.uint8) * 255
        final_composite = cv2.vconcat([annotated_resized, margin_img, faces_composite])
    else:
        final_composite = annotated_resized
    
    return final_composite

def get_all_image_paths(folder: str) -> list:
    """
    Retrieve all image paths from the specified folder.
    
    Args:
        folder (str): Path to the folder containing images.
    
    Returns:
        list: List of image file paths.
    """
    possible_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
    image_paths = []
    for ext in possible_extensions:
        pattern = os.path.join(folder, ext)
        image_paths.extend(glob.glob(pattern))
    return image_paths