# scripts/test_image_processing.py

import os
import sys
import glob
import logging
import logging_config  # Ensure the global logging is configured
import cv2
import numpy as np
from app.image_processing.face_detector import FaceDetector
from app.image_processing.face_aligner import FaceAligner
from app.image_processing.face_encoder import FaceEncoder
from app.image_processing import preprocessing

logger = logging.getLogger(__name__)

def draw_detections(image: np.ndarray, boxes: np.ndarray, landmarks: list) -> np.ndarray:
    """
    Draw bounding boxes and landmarks on the image.
    
    Args:
        image (np.ndarray): Original image.
        boxes (np.ndarray): Array of bounding boxes.
        landmarks (list): List of landmarks corresponding to each detected face.
    
    Returns:
        np.ndarray: Annotated image.
    """
    annotated = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        # Draw a rectangle around the face
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw each landmark as a small circle
        if landmarks is not None and i < len(landmarks):
            for (x, y) in landmarks[i]:
                cv2.circle(annotated, (int(x), int(y)), 2, (0, 0, 255), -1)
    return annotated

def create_composite_image(annotated_img: np.ndarray, face_rows: list, target_width: int = 448) -> np.ndarray:
    """
    Combine the annotated detection image and the per-face rows into a single composite image.
    
    Args:
        annotated_img (np.ndarray): Annotated original image.
        face_rows (list): List of images (rows) each showing the cropped and aligned face.
        target_width (int): The width to which face rows are set (e.g., crop+aligned images concatenated).
    
    Returns:
        np.ndarray: The final composite image.
    """
    # Resize the annotated image to have the same width as the composite rows
    # Preserve the aspect ratio
    annotated_h, annotated_w = annotated_img.shape[:2]
    new_height = int(annotated_h * (target_width / annotated_w))
    annotated_resized = cv2.resize(annotated_img, (target_width, new_height))
    
    if face_rows:
        # Vertically stack all face rows
        faces_composite = cv2.vconcat(face_rows)
        # Optionally, add a margin (a horizontal line) between the annotated image and the face rows
        margin = 10
        margin_img = np.ones((margin, target_width, 3), dtype=np.uint8) * 255  # white margin
        final_composite = cv2.vconcat([annotated_resized, margin_img, faces_composite])
    else:
        final_composite = annotated_resized
    
    return final_composite

def run_full_pipeline(image_path: str, device: str = 'cpu'):
    """
    Process a single image through the full pipeline: reading, detection, cropping,
    alignment, and encoding. A composite image showing all steps is saved to the results folder.
    
    Args:
        image_path (str): Path to the input image.
        device (str): Device on which to run models (default 'cpu').
    """
    # Create a results folder if it doesn't exist
    results_folder = os.path.join(os.getcwd(), "results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Step 1: Read the image
    image = preprocessing.read_image(image_path)
    
    # Step 2: Detect faces in the image
    detector = FaceDetector(device=device)
    boxes, probs, landmarks = detector.detect_faces(image)
    if boxes is None:
        logger.error("No faces detected in the image: %s", image_path)
        return
    
    # Draw detections (boxes & landmarks) on a copy of the original image
    detected_image = draw_detections(image, boxes, landmarks)
    
    # Prepare a list to hold each face's composite row (cropped and aligned side by side)
    face_rows = []
    
    # Define a fixed size (e.g., 224x224) for both cropped and aligned faces
    fixed_size = (224, 224)
    
    # Initialize the aligner and encoder only once (for all faces)
    aligner = FaceAligner()
    encoder = FaceEncoder(device=device)
    
    # Process each detected face
    for i, (box, face_landmarks) in enumerate(zip(boxes, landmarks)):
        # Step 3: Crop the detected face from the original image
        x1, y1, x2, y2 = box.astype(int)
        face_crop = image[y1:y2, x1:x2]
        # Resize the cropped face to the fixed size
        cropped_resized = cv2.resize(face_crop, fixed_size)
        
        # Step 4: Align the face using detected landmarks
        aligned_face = aligner.align(image, face_landmarks)
        # Ensure the aligned face is resized to the fixed size
        aligned_face_resized = cv2.resize(aligned_face, fixed_size)
        
        # Create a row by horizontally concatenating the cropped and aligned images
        row = cv2.hconcat([cropped_resized, aligned_face_resized])
        face_rows.append(row)
        
        # Log the embedding for this face (optional)
        embedding = encoder.encode(aligned_face)
        logger.info("Embedding for face %d: %s", i+1, embedding)
    
    # Create the final composite image by stacking the annotated image on top of all face rows
    # The expected width of the composite rows is fixed_size[0] * 2 (since we have 2 images side by side)
    composite_width = fixed_size[0] * 2
    final_composite = create_composite_image(detected_image, face_rows, target_width=composite_width)
    
    # Use the input image's base name to create a unique composite filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    composite_filename = f"pipeline_result_{base_name}.jpg"
    composite_path = os.path.join(results_folder, composite_filename)
    cv2.imwrite(composite_path, final_composite)
    logger.info("Composite pipeline image saved to %s", composite_path)

def get_all_image_paths():
    """
    Get a list of all image paths from the "images" folder in the project root.
    Supports common image extensions.
    
    Returns:
        list: A list of image file paths.
    """
    default_folder = os.path.join(os.getcwd(), "images")
    possible_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
    image_paths = []
    for ext in possible_extensions:
        pattern = os.path.join(default_folder, ext)
        files = glob.glob(pattern)
        image_paths.extend(files)
    return image_paths

if __name__ == "__main__":
    # If an image path is provided via command-line, process that image only;
    # Otherwise, process all images found in the "images" folder.
    if len(sys.argv) > 1:
        image_paths = [sys.argv[1]]
    else:
        image_paths = get_all_image_paths()
        if not image_paths:
            logger.error("No image path provided and no images found in the 'images' folder.")
            sys.exit(1)
    
    for image_path in image_paths:
        logger.info("Processing image: %s", image_path)
        run_full_pipeline(image_path)
    
    logger.info("Full pipeline processing completed successfully for %d image(s).", len(image_paths))