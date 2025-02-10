"""
Module: celeba_ingestion.py

This module downloads (via Kaggle API) a subset of the CelebA dataset from Kaggle,
selects a subset of images from the attribute CSV (e.g. the first 100 images),
and then inserts documents for each image into MongoDB using the DBManager.

Each document will contain:
  - celebrity_id: (simulated as the image filename without extension)
  - name: Generated as "Celebrity_{image_name}"
  - age: A random integer between 20 and 60 (simulated)
  - image_url: Local file path to the image (constructed from the dataset directory)

This provides a proof-of-concept ingestion pipeline.
"""

import os
import logging
import random
import pandas as pd
from dotenv import load_dotenv

from app.data.db_manager import DBManager

# Load environment variables from .env file
load_dotenv()
import logging_config  # Ensure the configuration is loaded
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
    # This will download a zip file and unzip it into output_dir.
    api.dataset_download_files(dataset_id, path=output_dir, unzip=True)
    logger.info("Download and extraction complete. Files are in %s", output_dir)

def parse_attr_file(attr_file: str, images_folder: str, top_n: int = 100):
    """
    Parse the list_attr_celeba.csv file and return a list of celebrity records.
    
    We assume the CSV file has a header row. The first column is the image filename.
    For each row (image), we simulate the celebrity identity by taking the filename (without extension)
    as the celebrity_id. We also generate a name as "Celebrity_{filename}" and a random age.
    
    Args:
        attr_file (str): Path to list_attr_celeba.csv.
        images_folder (str): Path to the folder containing images (note: in your dataset, images are in
                             output_dir/img_align_celeba/img_align_celeba).
        top_n (int): Number of records to select.
    
    Returns:
        list: A list of dictionaries, each containing:
              - celebrity_id: (str)
              - name: (str)
              - age: (int)
              - image_url: (str) Local file path to one image.
    """
    try:
        df = pd.read_csv(attr_file)
    except Exception as e:
        logger.error("Error reading attributes file %s: %s", attr_file, e)
        raise e

    # The first column is assumed to be the image file name.
    # We'll simply select the first top_n rows.
    df_subset = df.head(top_n)
    records = []
    for _, row in df_subset.iterrows():
        filename = row.iloc[0]  # Assume first column is filename, e.g., "000001.jpg"
        # Remove extension to use as celebrity_id
        celebrity_id = os.path.splitext(filename)[0]
        record = {
            "celebrity_id": celebrity_id,
            "name": f"Celebrity_{celebrity_id}",
            "age": random.randint(20, 60),
            "image_url": os.path.join(images_folder, filename)
        }
        records.append(record)
    logger.info("Prepared %d celebrity records from the attributes file.", len(records))
    return records

def ingest_celebrities(dataset_id: str, output_dir: str, top_n: int = 100):
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
    
    # In your case, you mentioned that the folder structure is:
    # output_dir/img_align_celeba/img_align_celeba/ contains all the images.
    images_folder = os.path.join(output_dir, "img_align_celeba", "img_align_celeba")
    # Use list_attr_celeba.csv (the file may be in the output_dir)
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
    ingest_celebrities(dataset_id=DATASET_ID, output_dir=OUTPUT_DIR, top_n=100)