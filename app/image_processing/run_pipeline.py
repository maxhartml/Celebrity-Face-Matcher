"""
Module: run_pipeline.py

This module orchestrates the full image processing pipeline:
- Reads an image.
- Detects faces.
- Draws detections (bounding boxes and landmarks).
- Crops and aligns faces.
- Computes facial embeddings.
- Creates a composite output image for visualization.

The primary function, process_image(), returns the composite image and a list of embeddings.
"""

import os
import sys
import logging
import cv2
import numpy as np
import logging_config  # Assumes this is at the project root

from .face_detector import FaceDetector
from .face_aligner import FaceAligner
from .face_encoder import FaceEncoder
from . import utils

logger = logging.getLogger(__name__)

def process_image(image_path: str, device: str = 'cpu'):
    """
    Process a single image through the full pipeline.

    Steps:
      1. Read the image.
      2. Detect faces.
      3. Draw detections on a copy of the image.
      4. For each detected face:
         - Crop the face.
         - Align the face.
         - Compute the facial embedding.
         - Create a visual row (cropped face and aligned face side-by-side).
      5. Create a composite image that shows the annotated original image and all per-face rows.
      
    Args:
        image_path (str): Path to the input image.
        device (str): Device to run models on ('cpu' or 'cuda').

    Returns:
        tuple: (composite_image, list_of_embeddings)
            - composite_image: The final composite image (numpy array) for visualization.
            - list_of_embeddings: A list of facial embeddings (one per detected face).
    """
    # Create a results folder (for debugging visualization) if needed.
    results_folder = os.path.join(os.getcwd(), "play/results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Step 1: Read the image.
    image = utils.read_image(image_path)
    
    # Step 2: Detect faces.
    detector = FaceDetector(device=device)
    boxes, probs, landmarks = detector.detect_faces(image)
    if boxes is None:
        logger.error("No faces detected in image: %s", image_path)
        return None, []
    
    # Draw detections (boxes and landmarks) on a copy of the original image.
    detected_image = utils.draw_detections(image, boxes, landmarks)
    
    # Prepare a list for per-face composite rows and for embeddings.
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
        
        # Create a row by horizontally concatenating the cropped and aligned images.
        row = cv2.hconcat([cropped_resized, aligned_face_resized])
        face_rows.append(row)
        
        # Compute embedding for the aligned face.
        embedding = encoder.encode(aligned_face)
        embeddings.append(embedding)
        logger.info("Computed embedding for face %d.", i+1)
    
    # Create a composite image for visualization.
    composite_width = fixed_size[0] * 2  # Two images side-by-side per row.
    composite_image = utils.create_composite_image(detected_image, face_rows, target_width=composite_width)
    
    # Optionally, save the composite image for debugging.
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    composite_filename = f"pipeline_result_{base_name}.jpg"
    composite_path = os.path.join(results_folder, composite_filename)
    cv2.imwrite(composite_path, composite_image)
    logger.info("Composite pipeline image saved to %s", composite_path)
    
    return composite_image, embeddings

def main():
    """
    Main entry point for testing.
    
    If a single image path is provided via command-line, process that image;
    otherwise, process all images found in the designated test folder.
    """
    if len(sys.argv) > 1:
        image_paths = [sys.argv[1]]
    else:
        images_folder = os.path.join(os.getcwd(), "play/test_images")
        image_paths = utils.get_all_image_paths(images_folder)
        if not image_paths:
            logger.error("No image found in the test images folder.")
            sys.exit(1)
    
    for image_path in image_paths:
        logger.info("Processing image: %s", image_path)
        composite, embeddings = process_image(image_path)
        logger.info("Processed %s: %d face embeddings extracted.", image_path, len(embeddings))
    
    logger.info("Image processing complete.")

if __name__ == "__main__":
    main()