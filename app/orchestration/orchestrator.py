"""
Module: orchestrator.py

This module orchestrates the batch processing of celebrity images:
- Retrieves celebrity records from MongoDB.
- Downloads celebrity images.
- Processes each image through the image processing pipeline (using process_image_array() from run_pipeline).
- Saves both the original and composite images to a dedicated subfolder for each celebrity in "celebrity_images".
- Upserts the extracted embedding (for simplicity, the first embedding) along with the attributes metadata into the Pinecone vector store.
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
from app.vector_store import pinecone_client
from app.image_processing import run_pipeline

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger("app.orchestration.orchestrator")

def download_image(image_url: str) -> np.ndarray:
    """
    Download an image from a URL or load it from a local file path,
    converting it to an OpenCV BGR image.

    Args:
        image_url (str): The URL or local file path of the image.

    Returns:
        np.ndarray: The image in BGR format.
    """
    if image_url.startswith("http://") or image_url.startswith("https://"):
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
    else:
        try:
            image_cv = cv2.imread(image_url)
            if image_cv is None:
                raise ValueError(f"Local file at {image_url} could not be read.")
            logger.info("Loaded local image from path: %s", image_url)
            return image_cv
        except Exception as e:
            logger.error("Error reading local image from path %s: %s", image_url, e)
            raise e

def orchestrate_embeddings(device: str = 'cpu'):
    """
    Orchestrate the batch processing of celebrity images:
    - Retrieve celebrity records from MongoDB.
    - For each record:
        - Download the image.
        - Process the image via process_image_array() to extract embeddings.
        - Save both the original and composite images to a dedicated subfolder.
        - Upsert the first embedding (for simplicity) along with the attributes metadata into Pinecone.
    """
    try:
        db_manager = DBManager()
        celebrities = db_manager.get_all_documents()[:300]  # Limit to 10 records for testing
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

    processed_count = 0
    for celeb in celebrities:
        try:
            # Use celebrity_id as both the unique ID and name.
            logger.info("Processing celebrity record: %s", celeb.get("celebrity_id"))
            celebrity_id = str(celeb.get("celebrity_id"))
            name = celebrity_id
            image_url = celeb.get("image_url")
            attributes = celeb.get("attributes")  # This is now the metadata.
            if not image_url:
                logger.warning("Celebrity %s has no image URL; skipping.", celebrity_id)
                continue

            # Download or load the image.
            image = download_image(image_url)

            # Process the image using process_image_array().
            embeddings = run_pipeline.process_image_array(image=image, name=name, device=device)
            if not embeddings:
                logger.warning("No embeddings extracted for celebrity %s; skipping.", celebrity_id)
                continue

            # Use the first embedding for upsert.
            embedding = embeddings[0]

            # Prepare metadata from the attributes dictionary.
            metadata = attributes if attributes else {}

            # Upsert the embedding into Pinecone.
            pinecone_client.upsert_embedding(index, celebrity_id=celebrity_id, embedding=embedding, metadata=metadata)
            logger.info("Upserted embedding for celebrity %s.", celebrity_id)
            processed_count += 1

        except Exception as e:
            logger.error("Error processing celebrity record %s: %s", celeb.get("celebrity_id"), e)

    logger.info("Orchestration complete. Processed %d out of %d records.", processed_count, len(celebrities))

if __name__ == "__main__":
    orchestrate_embeddings(device='cpu')