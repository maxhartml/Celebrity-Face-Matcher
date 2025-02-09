import logging
import cv2
import numpy as np
from facenet_pytorch import MTCNN

# Setup logger for this module
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    FaceDetector class using MTCNN for detecting faces in images.
    
    This class leverages the facenet-pytorch MTCNN model to detect faces.
    It returns bounding boxes and facial landmarks for each detected face.
    """
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the FaceDetector.
        
        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        logger.info("Initializing FaceDetector on device: %s", device)
        self.detector = MTCNN(keep_all=True, device=device)
    
    def detect_faces(self, image: np.ndarray):
        """
        Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image (BGR or RGB).
        
        Returns:
            boxes (np.ndarray): Array of bounding boxes [x1, y1, x2, y2] for detected faces.
            probs (np.ndarray): Confidence scores for each detection.
            landmarks (list): List of landmarks (if available) for each detected face.
        """
        logger.debug("Detecting faces in image with shape: %s", image.shape)
        # Convert image to RGB if needed (MTCNN expects RGB images)
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        boxes, probs, landmarks = self.detector.detect(image_rgb, landmarks=True)
        
        if boxes is None:
            logger.info("No faces detected in the image.")
            return None, None, None
        
        logger.info("Detected %d face(s).", len(boxes))
        return boxes, probs, landmarks

# For standalone testing (optional)
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path is None:
        logger.error("No image path provided!")
    else:
        img = cv2.imread(image_path)
        fd = FaceDetector(device='cpu')
        boxes, probs, landmarks = fd.detect_faces(img)
        logger.info("Detected Boxes: %s", boxes)