"""
Module: run_pipeline.py

This module orchestrates the full image processing pipeline:
- Reading an image.
- Detecting faces.
- Drawing detections.
- Cropping and aligning faces.
- Creating a composite output image.
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
from . import utils  # Our new utilities module

logger = logging.getLogger(__name__)

def run_full_pipeline(image_path: str, device: str = 'cpu'):
    """
    Run the full processing pipeline on a single image.
    
    Args:
        image_path (str): Path to the input image.
        device (str): Device to run models on ('cpu' or 'cuda').
    """
    results_folder = os.path.join(os.getcwd(), "play/results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Step 1: Read the image.
    image = utils.read_image(image_path)
    
    # Step 2: Detect faces.
    detector = FaceDetector(device=device)
    boxes, probs, landmarks = detector.detect_faces(image)
    if boxes is None:
        logger.error("No faces detected in the image: %s", image_path)
        return
    
    # Draw detection results on a copy of the image.
    detected_image = utils.draw_detections(image, boxes, landmarks)
    
    # Prepare rows to display the cropped and aligned faces side by side.
    face_rows = []
    fixed_size = (224, 224)
    
    aligner = FaceAligner()
    encoder = FaceEncoder(device=device)
    
    for i, (box, face_landmarks) in enumerate(zip(boxes, landmarks)):
        x1, y1, x2, y2 = box.astype(int)
        face_crop = image[y1:y2, x1:x2]
        cropped_resized = cv2.resize(face_crop, fixed_size)
        
        aligned_face = aligner.align(image, face_landmarks)
        aligned_face_resized = cv2.resize(aligned_face, fixed_size)
        
        # Create a row by concatenating the cropped and aligned images.
        row = cv2.hconcat([cropped_resized, aligned_face_resized])
        face_rows.append(row)
        
        embedding = encoder.encode(aligned_face)
        logger.info("Embedding for face %d: %s", i+1, embedding)
    
    composite_width = fixed_size[0] * 2  # 2 images side by side.
    final_composite = utils.create_composite_image(detected_image, face_rows, target_width=composite_width)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    composite_filename = f"pipeline_result_{base_name}.jpg"
    composite_path = os.path.join(results_folder, composite_filename)
    cv2.imwrite(composite_path, final_composite)
    logger.info("Composite pipeline image saved to %s", composite_path)

def main():
    """
    Main entry point: if a single image path is provided as an argument, process that image;
    otherwise, process all images found in the 'images' folder.
    """
    if len(sys.argv) > 1:
        image_paths = [sys.argv[1]]
    else:
        images_folder = os.path.join(os.getcwd(), "play/test_images")
        image_paths = utils.get_all_image_paths(images_folder)
        if not image_paths:
            logger.error("No image path provided and no images found in the 'images' folder.")
            sys.exit(1)
    
    for image_path in image_paths:
        logger.info("Processing image: %s", image_path)
        run_full_pipeline(image_path)
    
    logger.info("Full pipeline processing completed successfully for %d image(s).", len(image_paths))

if __name__ == "__main__":
    main()