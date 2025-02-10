import logging
import cv2
import numpy as np

# Setup logger for this module
logger = logging.getLogger(__name__)

class FaceAligner:
    """
    FaceAligner class to align face images using facial landmarks.
    
    This class uses an affine transformation based on the positions of the eyes
    to align the face so that it is standardized for embedding extraction.
    """
    
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=224, desired_face_height=None):
        """
        Initialize the FaceAligner.
        
        Args:
            desired_left_eye (tuple): Desired normalized position (x, y) for the left eye.
            desired_face_width (int): The desired output face width.
            desired_face_height (int): The desired output face height. Defaults to face width if None.
        """
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height if desired_face_height is not None else desired_face_width
        logger.info("FaceAligner initialized with desired size: (%d, %d)", 
                    self.desired_face_width, self.desired_face_height)
    
    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align the face in the image based on the provided landmarks.
        
        Args:
            image (np.ndarray): The original image.
            landmarks (np.ndarray): Facial landmarks for the face in the order 
                                    [left_eye, right_eye, nose, mouth_left, mouth_right].
        
        Returns:
            np.ndarray: The aligned face image.
        """
        logger.debug("Starting face alignment using landmarks: %s", landmarks)
        try:
            # Extract the left and right eye coordinates
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Compute the angle between the eyes
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            logger.debug("Computed rotation angle: %.2f degrees", angle)
            
            # Compute desired right eye position
            desired_right_eye_x = 1.0 - self.desired_left_eye[0]
            
            # Calculate the distance between the eyes in the input image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.desired_face_width
            scale = desired_dist / dist
            logger.debug("Computed scale factor: %.2f", scale)
            
            # Compute the center of the eyes
            eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            logger.debug("Eyes center: %s", eyes_center)
            
            # Compute the affine transformation matrix for rotation and scaling
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
            tX = self.desired_face_width * 0.5
            tY = self.desired_face_height * self.desired_left_eye[1]
            M[0, 2] += (tX - eyes_center[0])
            M[1, 2] += (tY - eyes_center[1])
            
            # Apply the affine transformation
            aligned_face = cv2.warpAffine(image, M, (self.desired_face_width, self.desired_face_height), flags=cv2.INTER_CUBIC)
        except Exception as e:
            logger.error("Error during face alignment: %s", e, exc_info=True)
            raise e
        
        logger.info("Face alignment completed.")
        return aligned_face