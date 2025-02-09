# scripts/test_image_processing.py

import logging_config  # Ensure the global logging is configured
import cv2
import sys
import logging
from app.image_processing.face_detector import FaceDetector
from app.image_processing.face_aligner import FaceAligner
from app.image_processing.face_encoder import FaceEncoder
from app.image_processing import preprocessing

logger = logging.getLogger(__name__)

def run_full_pipeline(image_path: str, device: str = 'cpu'):
    # Step 1: Read the image
    image = preprocessing.read_image(image_path)
    
    # Step 2: Detect faces in the image
    detector = FaceDetector(device=device)
    boxes, probs, landmarks = detector.detect_faces(image)
    if boxes is None:
        logger.error("No faces detected in the image.")
        return

    # For demonstration, pick the first detected face
    first_box = boxes[0]
    first_landmarks = landmarks[0]
    
    # Step 3: Crop the detected face
    x1, y1, x2, y2 = first_box.astype(int)
    face_crop = image[y1:y2, x1:x2]
    
    # Step 4: Align the face using detected landmarks
    aligner = FaceAligner()
    aligned_face = aligner.align(image, first_landmarks)
    
    # Step 5: Encode the aligned face
    encoder = FaceEncoder(device=device)
    embedding = encoder.encode(aligned_face)
    
    # Log the resulting embedding
    logger.info("Embedding for the detected face: %s", embedding)

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path is None:
        logger.error("Usage: python test_image_processing.py <path_to_image>")
    else:
        run_full_pipeline(image_path)
        logging.info("Full pipeline completed successfully.")