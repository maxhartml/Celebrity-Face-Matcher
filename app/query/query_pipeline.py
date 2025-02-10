"""
Module: query_pipeline.py

This module provides functionality to query the Pinecone vector store using an image.
It performs the following steps:
  1. Loads an image from a file path.
  2. Processes the image through the image processing pipeline to extract facial embeddings.
  3. Uses the first embedding as the query vector.
  4. Queries the Pinecone index for the top 5 similar vectors.
  5. Composes a new composite image that includes the query image and the top match images.
  6. Saves the composite image in a designated results folder.
  
This module is placed in the 'app/query' package to clearly separate querying/inference logic 
from ingestion and vector store operations.
"""

import os
import sys
import logging
import cv2
import numpy as np
import logging_config  # Ensure central logging is configured
from dotenv import load_dotenv

# Import our Pinecone functions from the vector store package.
from app.vector_store import pinecone_client
# Import our image processing pipeline and utility functions.
from app.image_processing import run_pipeline
from app.image_processing import utils

# Load environment variables.
load_dotenv()

logger = logging.getLogger("app.query.query_pipeline")

def query_image(image_path: str, device: str = 'cpu', top_k: int = 5) -> list:
    """
    Process an image to extract its facial embedding and then query the Pinecone index 
    for the top_k similar vectors.

    Args:
        image_path (str): Path to the image file.
        device (str): Device to run models on ('cpu' or 'cuda').
        top_k (int): Number of top matches to return.

    Returns:
        list: A list of matching vectors from Pinecone (each including id, score, and metadata).
    """
    try:
        logger.info("Querying image: %s", image_path)
        
        # Process the image using the image processing pipeline.
        # Use process_image() which reads the image from disk.
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        embeddings = run_pipeline.process_image(image_path, device=device, name=base_name)
        
        if not embeddings:
            logger.error("No embeddings extracted from image %s", image_path)
            return []
        
        # Convert the first embedding (if it is a numpy array) to a list.
        query_vector = embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else embeddings[0]
        logger.info("Extracted query embedding for image %s.", image_path)
        
        # Initialize (or connect to) the Pinecone index.
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            logger.error("PINECONE_INDEX_NAME is not set in the environment.")
            return []
        index = pinecone_client.initialize_index(index_name=index_name, dimension=512)
        
        # Query the index using the query vector.
        matches = pinecone_client.query_embedding(index, query_vector, top_k=top_k, include_metadata=True)
        logger.info("Query returned %d matches.", len(matches))
        return matches
    except Exception as e:
        logger.error("Error querying image %s: %s", image_path, e)
        return []

def compose_query_results(query_image_path: str, matches: list, output_folder: str, composite_filename: str, target_height: int = 224) -> str:
    """
    Compose a composite image that includes the query image and the top matching images
    arranged horizontally. The query image appears on the left and the match images are 
    placed to its right.

    Args:
        query_image_path (str): Path to the query image.
        matches (list): A list of match dictionaries from Pinecone query. Each match should have 
                        a "metadata" dictionary that includes either 'saved_original' or 'image_url'.
        output_folder (str): The folder where the composite image will be saved.
        composite_filename (str): The filename for the composite image.
        target_height (int): The height to which each image will be resized.

    Returns:
        str: The full path to the saved composite image.
    """
    # Load the query image.
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        raise ValueError(f"Could not load query image from {query_image_path}")
    
    # Resize query image to the target height while maintaining aspect ratio.
    q_h, q_w = query_img.shape[:2]
    scale = target_height / q_h
    query_resized = cv2.resize(query_img, (int(q_w * scale), target_height))
    
    # For each match, attempt to load the image.
    match_images = []
    for match in matches:
        metadata = match.get("metadata", {})
        # Prefer the saved original if available, else fall back to image_url.
        match_img_path = metadata.get("saved_original") or metadata.get("image_url")
        if not match_img_path:
            logger.warning("No image path found in metadata for match with ID %s.", match.get("id"))
            continue
        match_img = cv2.imread(match_img_path)
        if match_img is None:
            logger.warning("Could not load match image from %s", match_img_path)
            continue
        # Resize match image to target height.
        m_h, m_w = match_img.shape[:2]
        scale = target_height / m_h
        match_resized = cv2.resize(match_img, (int(m_w * scale), target_height))
        match_images.append(match_resized)
    
    if not match_images:
        raise ValueError("No valid match images could be loaded.")
    
    # Concatenate query image and match images horizontally.
    composite_image = cv2.hconcat([query_resized] + match_images)
    
    # Save the composite image using the utility function.
    saved_path = utils.save_image(composite_image, output_folder, composite_filename)
    logger.info("Composite query result image saved to %s", saved_path)
    return saved_path

def main():
    """
    Main entry point for testing the query pipeline.
    
    If an image path is provided via the command line, that image is used.
    Otherwise, the first image from the "images/test_images" folder is used.
    Then the image is processed, queried against Pinecone, and a composite result image is created.
    """
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        test_folder = os.path.join(os.getcwd(), "images/test_images")
        paths = utils.get_all_image_paths(test_folder)
        if not paths:
            logger.error("No images found in the test images folder: %s", test_folder)
            sys.exit(1)
        image_path = paths[2]
    
    # Query the Pinecone index using the chosen image.
    matches = query_image(image_path, device='cpu', top_k=5)
    if matches:
        logger.info("Top %d matches for image %s:", len(matches), image_path)
        for match in matches:
            logger.info("ID: %s, Score: %.4f, Metadata: %s", match.get("id"), match.get("score"), match.get("metadata"))
    else:
        logger.error("No matches found for image %s", image_path)
        sys.exit(1)
    
    # Compose a composite result image from the query image and its top matches.
    results_folder = os.path.join(os.getcwd(), "images/results")
    composite_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_query_results.jpg"
    try:
        composite_result_path = compose_query_results(image_path, matches, results_folder, composite_filename, target_height=224)
        logger.info("Composite query result saved at: %s", composite_result_path)
    except Exception as e:
        logger.error("Error composing query results: %s", e)

if __name__ == "__main__":
    main()