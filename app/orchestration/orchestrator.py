"""
Module: orchestrator.py

This module orchestrates the batch processing of celebrity images:
- Retrieves celebrity records from MongoDB.
- Downloads celebrity images.
- Processes each image through the image processing pipeline (using process_image_array() from run_pipeline).
- Saves both the original and composite images to a dedicated subfolder.
- Upserts the extracted embedding (for simplicity, the first embedding) along with combined metadata
  (flattened attributes and image_url) into the Pinecone vector store using the new PineconeClient class.
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
from app.config import *
from app.data.db_manager import DBManager
from app.image_processing import run_pipeline
from app.image_processing import utils
# Import the new PineconeClient class
from app.vector_store.pinecone_client import PineconeClient

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
            logger.error("Error downloading image from URL %s: %s", image_url, e, exc_info=True)
            raise e
    else:
        try:
            image_cv = cv2.imread(image_url)
            if image_cv is None:
                raise ValueError(f"Local file at {image_url} could not be read.")
            logger.info("Loaded local image from path: %s", image_url)
            return image_cv
        except Exception as e:
            logger.error("Error reading local image from path %s: %s", image_url, e, exc_info=True)
            raise e

def orchestrate_embeddings():
    """
    Orchestrate the batch processing of celebrity images:
    - Retrieve celebrity records from MongoDB.
    - For each record:
        - Download the image.
        - Process the image via process_image_array() to extract embeddings.
        - Save both the original and composite images to a dedicated subfolder.
        - Upsert the first embedding (for simplicity) along with a flattened metadata dictionary
          (merging attributes and image_url) into Pinecone.
    """
    try:
        db_manager = DBManager()
        # For testing purposes, limit to 200 records (adjust as needed)
        # celebrities = db_manager.get_all_documents()
        celebrities = db_manager.get_all_documents(limit=200)
        if not celebrities:
            logger.error("No celebrity records found in the database.")
            return
        logger.info("Found %d celebrity records.", len(celebrities))
    except Exception as e:
        logger.error("Error retrieving celebrity records: %s", e, exc_info=True)
        return

    # Instantiate the new PineconeClient.
    try:
        pc_client = PineconeClient()  # Uses hardcoded values from your environment/config
        logger.info("Connected to Pinecone index '%s'.", pc_client.index_name)
    except Exception as e:
        logger.error("Error initializing Pinecone client: %s", e, exc_info=True)
        return

    processed_count = 0
    total_records = len(celebrities)
    for celeb in celebrities:
        try:
            celebrity_id = str(celeb.get("celebrity_id"))
            logger.info("Processing celebrity record: %s", celebrity_id)
            # Use celebrity_id as both the unique ID and the name.
            name = celebrity_id  
            image_url = celeb.get("image_url")
            attributes = celeb.get("attributes")  # Enriched attribute data.

            if not image_url:
                logger.warning("Celebrity %s has no image URL; skipping.", celebrity_id)
                continue

            # Download or load the image.
            image = download_image(image_url)

            # Process the image using process_image_array().
            embeddings = run_pipeline.process_image_array(image=image, name=name, device=DEVICE)
            if not embeddings or embeddings[0] is None:
                logger.warning("No valid embedding extracted for celebrity %s; skipping.", celebrity_id)
                continue

            embedding = embeddings[0]
            logger.info("Extracted embedding for celebrity %s.", celebrity_id)

            # Prepare metadata: flatten the attributes dictionary into a flat metadata dictionary.
            metadata = {"image_url": image_url}
            if attributes and isinstance(attributes, dict):
                for key, value in attributes.items():
                    metadata[f"attr_{key}"] = value

            # Upsert the embedding into Pinecone using the new PineconeClient instance.
            try:
                upsert_result = pc_client.upsert_embedding(celebrity_id, embedding, metadata=metadata)
                if upsert_result is None or ("upsertedCount" in upsert_result and upsert_result["upsertedCount"] < 1):
                    logger.error("Upsert failed for celebrity %s; skipping.", celebrity_id)
                    continue
            except Exception as upsert_e:
                logger.error("Error during upsert for celebrity %s: %s", celebrity_id, upsert_e, exc_info=True)
                continue

            logger.info("Upserted embedding for celebrity %s.", celebrity_id)
            processed_count += 1

        except Exception as e:
            logger.error("Error processing celebrity record %s: %s", celeb.get("celebrity_id"), e, exc_info=True)

    logger.info("Orchestration complete. Processed %d out of %d records.", processed_count, total_records)

if __name__ == "__main__":
    orchestrate_embeddings()