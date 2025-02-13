"""
Module: run_pipeline.py

This module orchestrates the full image processing pipeline:
- Processes an image (either by file path or directly as a NumPy array).
- Detects faces.
- Draws detections (bounding boxes and landmarks).
- Crops and aligns faces.
- Computes facial embeddings.
- Creates a composite output image for visualization.
- Saves processed images in a dedicated subfolder if needed.

Primary functions:
  - process_image_array(image, device): Processes an image provided as a NumPy array.
  - process_image(image_path, device): Reads an image from disk and processes it.
"""

import os
import sys
import logging
import cv2
import numpy as np
import app.logging_config as logging_config  # Assumes this is at the project root
from .face_detector import FaceDetector
from .face_aligner import FaceAligner
from .face_encoder import FaceEncoder
from . import utils

logger = logging.getLogger(__name__)

def process_image_array(image: np.ndarray, name: str, device: str = 'cpu'):
    """
    Process a single image (as a NumPy array) through the full pipeline.

    Steps:
      1. Detect faces.
      2. Draw detections on a copy of the image.
      3. For each detected face:
         - Crop the face.
         - Align the face.
         - Compute the facial embedding.
         - Create a visual row (cropped face and aligned face side-by-side).
      4. Create a composite image that shows the annotated original image and all per-face rows.
      
    Args:
        image (np.ndarray): The input image in BGR format.
        device (str): Device to run models on ('cpu' or 'cuda').

    Returns:
        tuple: (composite_image, list_of_embeddings)
            - composite_image: The final composite image (numpy array) for visualization.
            - list_of_embeddings: A list of facial embeddings (one per detected face).
    """
    detector = FaceDetector(device=device)
    boxes, probs, landmarks = detector.detect_faces(image)
    if boxes is None:
        logger.error("No faces detected in the provided image.")
        return None, []
    
    detected_image = utils.draw_detections(image, boxes, landmarks)
    face_rows = []
    embeddings = []
    fixed_size = (224, 224)
    
    aligner = FaceAligner()
    encoder = FaceEncoder(device=device)
    
    for i, (box, face_landmarks) in enumerate(zip(boxes, landmarks)):
        x1, y1, x2, y2 = box.astype(int)
        face_crop = image[y1:y2, x1:x2]
        cropped_resized = cv2.resize(face_crop, fixed_size)
        
        aligned_face = aligner.align(image, face_landmarks)
        aligned_face_resized = cv2.resize(aligned_face, fixed_size)
        
        row = cv2.hconcat([cropped_resized, aligned_face_resized])
        face_rows.append(row)
        
        embedding = encoder.encode(aligned_face)
        embeddings.append(embedding)
        logger.info("Computed embedding for face %d.", i+1)
    
    composite_width = fixed_size[0] * 2  # Two images side-by-side per row.
    composite_image = utils.create_composite_image(detected_image, face_rows, target_width=composite_width)
    
    # Save the composite image for debugging if desired.
    results_folder = os.path.join(os.getcwd(), "images/processed_images")
    composite_filename = f"{name}_processed_image.jpg"  # Default name when no filename is provided.
    utils.save_image(composite_image, results_folder, composite_filename)
    processed_image_path = os.path.join(results_folder, composite_filename)
    logger.info("Saved processed image to %s.", processed_image_path)
    # return the first facial embedding
    return embeddings[0], probs[0], processed_image_path

def process_image(image_path: str, name: str, device: str = 'cpu'):
    """
    Process a single image from disk through the full pipeline.

    Args:
        image_path (str): Path to the image file.
        device (str): Device to run models on ('cpu' or 'cuda').

    Returns:
        list: A list of facial embeddings (one per detected face).
    """
    image = utils.read_image(image_path)
    if image is None:
        logger.error("Failed to read image from %s.", image_path)
        return []
    
    return process_image_array(image, name, device)

def main():
    """
    Main entry point for testing.
    If a single image path is provided via command-line, process that image;
    otherwise, process all images found in the designated test folder.
    """
    if len(sys.argv) > 1:
        image_paths = [sys.argv[1]]
    else:
        images_folder = os.path.join(os.getcwd(), "images/test_images")
        image_paths = utils.get_all_image_paths(images_folder)
        if not image_paths:
            logger.error("No image found in the test images folder.")
            sys.exit(1)
    
    for image_path in image_paths:
        logger.info("Processing image: %s", image_path)
        image_name = image_path.split('/')[-1].split('.')[0]
        embeddings = process_image(image_path=image_path, name=image_name)
        logger.info("Processed %s: %d face embeddings extracted.", image_path, len(embeddings))

    logger.info("Image processing complete.")

if __name__ == "__main__":
    main()