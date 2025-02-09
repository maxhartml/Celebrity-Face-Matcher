"""
Module: orchestrator.py

This module orchestrates the process of:
- Retrieving celebrity records from MongoDB.
- Downloading celebrity images.
- Processing each image through the image processing pipeline (using process_image() from run_pipeline).
- Upserting the extracted embedding(s) along with celebrity metadata into the Pinecone vector store.
"""

import os
import sys
import logging
import requests
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

from dotenv import load_dotenv
from app.data.db_manager import DBManager
from app.vector_space import pinecone_client
from app.image_processing import run_pipeline

load_dotenv()
logger = logging.getLogger(__name__)

def download_image(image_url: str) -> np.ndarray:
    """
    Download an image from a URL and convert it to an OpenCV BGR image.

    Args:
        image_url (str): The URL of the image.

    Returns:
        np.ndarray: The image in BGR format.
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_pil = Image.open(BytesIO(response.content)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        logger.info("Downloaded image from URL: %s", image_url)
        return image_cv
    except Exception as e:
        logger.error("Error downloading image from URL %s: %s", image_url, e)
        raise e

def orchestrate_embeddings(device: str = 'cpu'):
    """
    Orchestrate the batch processing of celebrity images:
    - Retrieve celebrity records from MongoDB.
    - For each record, download the image and save it temporarily.
    - Process the image via run_pipeline.process_image() to extract embeddings.
    - Upsert the first embedding (for simplicity) along with metadata into Pinecone.
    """
    try:
        db_manager = DBManager()
        celebrities = db_manager.get_all_documents()
        if not celebrities:
            logger.error("No celebrity records found in the database.")
            return
        logger.info("Found %d celebrity records.", len(celebrities))
    except Exception as e:
        logger.error("Error retrieving celebrity records: %s", e)
        return

    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        index = pinecone_client.initialize_index(index_name=index_name, dimension=512)
    except Exception as e:
        logger.error("Error initializing Pinecone index: %s", e)
        return

    temp_folder = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_folder, exist_ok=True)

    processed_count = 0
    for celeb in celebrities:
        try:
            celebrity_id = str(celeb.get("_id"))
            name = celeb.get("name", "Unknown")
            image_url = celeb.get("image_url")
            if not image_url:
                logger.warning("Celebrity %s (%s) does not have an image URL; skipping.", celebrity_id, name)
                continue

            # Download the image.
            image = download_image(image_url)
            temp_filename = f"temp_{celebrity_id}.jpg"
            temp_path = os.path.join(temp_folder, temp_filename)
            cv2.imwrite(temp_path, image)

            # Process the image to obtain embeddings.
            _, embeddings = run_pipeline.process_image(temp_path, device=device)
            if not embeddings:
                logger.warning("No embeddings extracted for celebrity %s (%s); skipping.", celebrity_id, name)
                continue

            # For simplicity, take the first embedding.
            embedding = embeddings[0]
            metadata = {
                "name": name,
                "biography": celeb.get("biography", ""),
                "image_url": image_url
            }

            # Upsert the embedding into Pinecone.
            pinecone_client.upsert_embedding(index, celebrity_id=celebrity_id, embedding=embedding, metadata=metadata)
            logger.info("Upserted embedding for celebrity %s (%s).", celebrity_id, name)
            processed_count += 1

            # Clean up the temporary file.
            os.remove(temp_path)

        except Exception as e:
            logger.error("Error processing celebrity record %s: %s", celeb.get("_id"), e)

    logger.info("Orchestration complete. Processed %d out of %d records.", processed_count, len(celebrities))

if __name__ == "__main__":
    orchestrate_embeddings(device='cpu')