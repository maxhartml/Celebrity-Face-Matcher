"""
Module: celeba_ingestion.py

This module downloads (via the Kaggle API) a subset of the CelebA dataset from Kaggle,
selects a subset of images from the attribute CSV (e.g., the first 100 images), and then
inserts documents for each image into MongoDB using DBManager.

Each document will contain:
  - celebrity_id: Extracted from the image filename (without extension)
  - image_url: Local file path to the image (constructed from the dataset directory)
  - attributes: A dictionary containing all attribute values from list_attr_celeba.csv

This provides a proof-of-concept ingestion pipeline with enriched metadata.
"""

import os
import logging
import pandas as pd
from dotenv import load_dotenv

from app.data.db_manager import DBManager

# Load environment variables from .env file
load_dotenv()
import logging_config  # Ensure the logging configuration is loaded
logger = logging.getLogger("app.data.celeba_ingestion")

def download_celeba_subset(dataset_id: str, output_dir: str):
    """
    Download and unzip the CelebA dataset from Kaggle using the Kaggle API.

    Args:
        dataset_id (str): The Kaggle dataset identifier (e.g., "jessicali9530/celeba-dataset").
        output_dir (str): The directory where the dataset should be downloaded and extracted.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading CelebA dataset '%s' from Kaggle...", dataset_id)
    os.makedirs(output_dir, exist_ok=True)
    # This downloads a zip file and unzips it into output_dir.
    api.dataset_download_files(dataset_id, path=output_dir, unzip=True)
    logger.info("Download and extraction complete. Files are in %s", output_dir)

def parse_attr_file(attr_file: str, images_folder: str, top_n: int):
    """
    Parse the list_attr_celeba.csv file and return a list of celebrity records.

    The CSV file is assumed to have a header row with columns such as:
      image_id, 5_o_Clock_Shadow, Arched_Eyebrows, Attractive, ..., Young

    For each row, the function:
      - Extracts the filename from the "image_id" column.
      - Derives the celebrity_id by removing the file extension.
      - Constructs the image_url using images_folder and the filename.
      - Stores all attribute values (all columns except "image_id") into an "attributes" dictionary.

    Args:
        attr_file (str): Path to list_attr_celeba.csv.
        images_folder (str): Path to the folder containing images (e.g., output_dir/img_align_celeba/img_align_celeba).
        top_n (int): Number of records to select.

    Returns:
        list: A list of dictionaries, each containing:
              - celebrity_id (str)
              - image_url (str)
              - attributes (dict)
    """
    try:
        df = pd.read_csv(attr_file)
    except Exception as e:
        logger.error("Error reading attributes file %s: %s", attr_file, e)
        raise e
    
    if top_n == -1:
        top_n = len(df)

    # Select the first top_n rows.
    df_subset = df.head(top_n)
    records = []
    for _, row in df_subset.iterrows():
        filename = row["image_id"]  # Expecting a column named "image_id", e.g., "000001.jpg"
        celebrity_id = os.path.splitext(filename)[0]
        # Create an attributes dictionary from all columns except 'image_id'
        attributes = row.drop("image_id").to_dict()
        record = {
            "celebrity_id": celebrity_id,
            "image_url": os.path.join(images_folder, filename),
            "attributes": attributes
        }
        records.append(record)
    logger.info("Prepared %d celebrity records from the attributes file.", len(records))
    return records

def ingest_celebrities(dataset_id: str, output_dir: str, top_n: int):
    """
    Download a subset of the CelebA dataset from Kaggle (if not already downloaded),
    parse the attribute CSV file to select top N records, and insert these celebrity records
    into MongoDB using DBManager.

    Args:
        dataset_id (str): Kaggle dataset identifier (e.g., "jessicali9530/celeba-dataset").
        output_dir (str): Directory where the dataset will be downloaded/extracted.
        top_n (int): Number of records to ingest.

    Returns:
        list: List of inserted document IDs.
    """
    # Download the dataset if necessary.
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        download_celeba_subset(dataset_id, output_dir)
    else:
        logger.info("Dataset already exists at %s", output_dir)
    
    # The expected folder structure:
    # output_dir/img_align_celeba/img_align_celeba/ contains all the images.
    images_folder = os.path.join(output_dir, "img_align_celeba", "img_align_celeba")
    # The attribute file is expected to be in output_dir.
    attr_file = os.path.join(output_dir, "list_attr_celeba.csv")
    
    if not os.path.exists(images_folder) or not os.path.exists(attr_file):
        logger.error("Expected dataset files not found in %s", output_dir)
        return

    records = parse_attr_file(attr_file, images_folder, top_n=top_n)

    try:
        db_manager = DBManager()
        inserted_ids = db_manager.insert_documents(records)
        logger.info("Ingested %d celebrity records into the database.", len(inserted_ids))
        return inserted_ids
    except Exception as e:
        logger.error("Error ingesting celebrity data: %s", e)
        raise e

if __name__ == "__main__":
    DATASET_ID = "jessicali9530/celeba-dataset"  # Replace if needed.
    OUTPUT_DIR = os.path.join(os.getcwd(), "celeba_data")
    ingest_celebrities(dataset_id=DATASET_ID, output_dir=OUTPUT_DIR, top_n=-1)