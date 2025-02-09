import logging
import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1

# Setup logger for this module
logger = logging.getLogger(__name__)

class FaceEncoder:
    """
    FaceEncoder class for generating facial embeddings.
    
    This class uses a pre-trained InceptionResnetV1 model (with VGGFace2 weights) to generate a 512-dimensional 
    embedding from an aligned face image. These embeddings are useful for comparing faces via similarity search.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the FaceEncoder.
        
        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        logger.info("Initializing FaceEncoder on device: %s", device)
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the aligned face image before encoding.
        
        Converts the image from BGR to RGB, resizes it to 160x160, and normalizes the pixel values.
        
        Args:
            face_image (np.ndarray): The aligned face image in BGR format.
        
        Returns:
            torch.Tensor: A tensor of shape (1, 3, 160, 160) ready for embedding extraction.
        """
        logger.debug("Preprocessing face image with shape: %s", face_image.shape)
        # Convert from BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # Resize to 160x160 (model expected input size)
        face_resized = cv2.resize(face_rgb, (160, 160))
        # Normalize pixel values to [0, 1]
        face_normalized = face_resized / 255.0
        # Convert to float32 and reorder dimensions to (C, H, W)
        face_tensor = torch.tensor(face_normalized, dtype=torch.float32).permute(2, 0, 1)
        # Scale to [-1, 1] (InceptionResnetV1 expects input in this range)
        face_tensor = (face_tensor - 0.5) / 0.5
        # Add batch dimension and move to the appropriate device
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        logger.debug("Face tensor shape after preprocessing: %s", face_tensor.shape)
        return face_tensor
    
    def encode(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate an embedding for the provided aligned face image.
        
        Args:
            face_image (np.ndarray): The aligned face image.
        
        Returns:
            np.ndarray: A 512-dimensional embedding vector.
        """
        logger.info("Encoding face image.")
        preprocessed_face = self.preprocess_face(face_image)
        with torch.no_grad():
            embedding = self.model(preprocessed_face)
        embedding_np = embedding.cpu().numpy().flatten()
        logger.info("Generated embedding of shape: %s", embedding_np.shape)
        logger.info("Embedding: %s", embedding_np)
        return embedding_np