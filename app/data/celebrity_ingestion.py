"""
Module: celebrity_ingestion.py

This module fetches celebrity images and metadata from an external API
and stores the retrieved data in MongoDB using the DBManager.
"""

import os
import logging
import requests
from dotenv import load_dotenv
from .db_manager import DBManager

# Load environment variables (if not already loaded by a higher-level script)
load_dotenv()

logger = logging.getLogger(__name__)

def fetch_celebrities_from_api(api_url: str, params: dict = None):
    """
    Fetch celebrity data from the API.
    
    Args:
        api_url (str): The API endpoint URL.
        params (dict): Optional parameters for the API request.
    
    Returns:
        list: A list of celebrity metadata dictionaries.
    """
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raises an HTTPError if the response code is not 200
        data = response.json()
        # Assume the API returns a JSON array of celebrity dictionaries.
        logger.info("Fetched %d celebrities from API.", len(data))
        return data
    except Exception as e:
        logger.error("Error fetching data from API: %s", e)
        raise e

def ingest_celebrities(api_url: str, params: dict = None):
    """
    Fetch celebrity data from the API and insert it into MongoDB.
    
    Args:
        api_url (str): The API endpoint URL.
        params (dict): Optional parameters for the API request.
    
    Returns:
        list: List of inserted document IDs.
    """
    db_manager = DBManager()
    celebrities = fetch_celebrities_from_api(api_url, params)
    # Here you might preprocess or validate each celebrity record as needed.
    try:
        inserted_ids = db_manager.insert_documents(celebrities)
        logger.info("Ingested %d celebrity records into the database.", len(inserted_ids))
        return inserted_ids
    except Exception as e:
        logger.error("Error ingesting celebrity data: %s", e)
        raise e

if __name__ == "__main__":
    # For testing purposes, the API URL can be provided via an environment variable or hardcoded.
    API_URL = os.getenv("CELEB_API_URL", "https://example.com/api/celebrities")
    params = {}  # Customize API parameters if needed.
    ingest_celebrities(API_URL, params)