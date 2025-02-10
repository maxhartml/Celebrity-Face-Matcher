"""
Module: query_pipeline.py

This module provides functionality to query the Pinecone vector store using an image.
It performs the following steps:
  1. Loads an image from a file path.
  2. Processes the image through the image processing pipeline to extract facial embeddings.
  3. Uses the first embedding as the query vector.
  4. Queries the Pinecone index for the top 5 similar vectors.
  5. Composes a composite image that includes the query image and the top match images.
  6. Saves the composite image in a designated results folder.
  7. Exports the match results (image_id, score, and attributes) to a CSV file, including a total row.
  
This module is placed in the 'app/query' package to clearly separate querying/inference logic 
from ingestion and vector store operations.
"""

import os
import sys
import csv
import logging
import cv2
import numpy as np
import logging_config  # Ensure central logging is configured
from dotenv import load_dotenv

from app.vector_store import pinecone_client
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
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # process_image() returns a tuple (composite_image, embeddings)
        embeddings = run_pipeline.process_image(image_path, device=device, name=base_name)
        
        if not embeddings or embeddings[0] is None:
            logger.error("No embeddings extracted from image %s", image_path)
            return []
        
        # Convert the first embedding to a list if needed.
        query_vector = embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else embeddings[0]
        logger.info("Extracted query embedding for image %s.", image_path)
        
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            logger.error("PINECONE_INDEX_NAME is not set in the environment.")
            return []
        index = pinecone_client.initialize_index(index_name=index_name, dimension=512)
        
        matches = pinecone_client.query_embedding(index, query_vector, top_k=top_k, include_metadata=True)
        logger.info("Query returned %d matches.", len(matches))
        return matches
    except Exception as e:
        logger.error("Error querying image %s: %s", image_path, e, exc_info=True)
        return []

def compose_query_results(query_image_path: str, matches: list, output_folder: str, composite_filename: str, target_height: int = 224) -> str:
    """
    Compose a composite image that includes the query image and the top matching images
    arranged horizontally. The query image appears on the left and the match images are 
    placed to its right.

    Args:
        query_image_path (str): Path to the query image.
        matches (list): A list of match dictionaries from the Pinecone query. Each match's metadata
                        should include an "image_url" field.
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
    
    # Resize the query image while maintaining its aspect ratio.
    q_h, q_w = query_img.shape[:2]
    scale = target_height / q_h
    query_resized = cv2.resize(query_img, (int(q_w * scale), target_height))
    
    match_images = []
    for match in matches:
        metadata = match.get("metadata", {})
        match_img_path = metadata.get("image_url")
        if not match_img_path:
            logger.warning("No image path found in metadata for match with ID %s.", match.get("id"))
            continue
        match_img = cv2.imread(match_img_path)
        if match_img is None:
            logger.warning("Could not load match image from %s", match_img_path)
            continue
        m_h, m_w = match_img.shape[:2]
        scale = target_height / m_h
        match_resized = cv2.resize(match_img, (int(m_w * scale), target_height))
        match_images.append(match_resized)
    
    if not match_images:
        raise ValueError("No valid match images could be loaded.")
    
    composite_image = cv2.hconcat([query_resized] + match_images)
    
    saved_path = utils.save_image(composite_image, output_folder, composite_filename)
    logger.info("Composite query result image saved to %s", saved_path)
    return saved_path

def export_matches_to_csv(matches: list, csv_filepath: str) -> str:
    """
    Export the query results (matches) to a CSV file with columns for image_id, score,
    and each attribute from the metadata. A total row is appended at the bottom that sums 
    up the attribute values across all matches.

    Args:
        matches (list): A list of match dictionaries from Pinecone query.
        csv_filepath (str): The full path to the CSV file to be created.

    Returns:
        str: The CSV filepath.
    """
    # Determine the set of attribute keys across all matches that start with "attr_"
    attribute_keys = set()
    for match in matches:
        metadata = match.get("metadata", {})
        for key in metadata.keys():
            # If you stored attributes without the "attr_" prefix, adjust accordingly.
            if key != "image_url":  # Exclude image_url from attribute aggregation.
                attribute_keys.add(key)
    attribute_keys = sorted(attribute_keys)
    
    headers = ["image_id", "score"] + attribute_keys
    rows = []
    totals = { key: 0 for key in attribute_keys }
    
    for match in matches:
        row = {}
        row["image_id"] = match.get("id", "")
        row["score"] = match.get("score", "")
        metadata = match.get("metadata", {})
        for key in attribute_keys:
            # If an attribute is missing, default to 0.
            value = metadata.get(key, 0)
            row[key] = value
            try:
                totals[key] += float(value)
            except Exception:
                totals[key] += 0
        rows.append(row)
    
    # Write to CSV file.
    try:
        with open(csv_filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            # Append the totals row.
            totals_row = {"image_id": "Total", "score": ""}
            totals_row.update(totals)
            writer.writerow(totals_row)
        logger.info("Exported query results to CSV file: %s", csv_filepath)
    except Exception as e:
        logger.error("Error exporting query results to CSV: %s", e, exc_info=True)
        raise e

    return csv_filepath

def main():
    """
    Main entry point for the query pipeline.
    
    This module must be called with an image path as a command-line argument.
    For example:
        python -m app.query.query_pipeline /path/to/your/test_image.jpg
    """
    if len(sys.argv) < 2:
        logger.error("No image path provided. Please run:\npython -m app.query.query_pipeline /path/to/your/test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    matches = query_image(image_path, device='cpu', top_k=20)
    if not matches:
        logger.error("No matches found for image %s", image_path)
        sys.exit(1)
    
    logger.info("Top %d matches for image %s:", len(matches), image_path)
    for match in matches:
        logger.info("ID: %s, Score: %.4f, Metadata: %s", match.get("id"), match.get("score"), match.get("metadata"))
    
    results_folder = os.path.join(os.getcwd(), "images/results")
    composite_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_query_results.jpg"
    csv_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_query_results.csv"
    
    try:
        composite_result_path = compose_query_results(image_path, matches, results_folder, composite_filename, target_height=224)
        logger.info("Composite query result saved at: %s", composite_result_path)
    except Exception as e:
        logger.error("Error composing query results: %s", e, exc_info=True)
        sys.exit(1)
    
    # Export matches to CSV.
    csv_filepath = os.path.join(results_folder, csv_filename)
    try:
        export_matches_to_csv(matches, csv_filepath)
    except Exception as e:
        logger.error("Error exporting query results to CSV: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()