"""
Module: pinecone_client.py

This module abstracts operations related to the Pinecone vector store.
It provides functions to initialize an index, upsert vectors, and query vectors.
"""

import os
import logging
from pinecone import Pinecone, ServerlessSpec  # New Pinecone client API
from dotenv import load_dotenv

# Load environment variables (if not already loaded)
load_dotenv()

logger = logging.getLogger(__name__)

def initialize_index(index_name: str = None, dimension: int = 512):
    """
    Initialize or connect to a Pinecone index using the new Pinecone client API.

    Args:
        index_name (str): The name of the index. If None, uses the PINECONE_INDEX_NAME env var.
        dimension (int): The dimensionality of the vectors (default is 512).

    Returns:
        Pinecone.Index: The Pinecone index object.
    """
    # Get API key from environment variables.
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("Pinecone API key not set in environment variables.")
        raise ValueError("Pinecone API key must be provided.")
    
    # Create an instance of the Pinecone client.
    pc = Pinecone(api_key=api_key)
    
    # Determine the index name.
    if index_name is None:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            logger.error("Pinecone index name not provided and PINECONE_INDEX_NAME env var is not set.")
            raise ValueError("Pinecone index name must be provided.")
    
    try:
        # List existing indexes.
        existing_indexes = pc.list_indexes()
        if index_name not in existing_indexes:
            logger.info("Index '%s' not found. Creating new index with dimension %d.", index_name, dimension)
            # Read cloud and region from environment variables (or use defaults)
            cloud = os.getenv("PINECONE_CLOUD")
            region = os.getenv("PINECONE_REGION")
            spec = ServerlessSpec(cloud=cloud, region=region)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",  # Using cosine similarity for this example.
                spec=spec
            )
        else:
            logger.info("Index '%s' already exists.", index_name)
    except Exception as e:
        logger.error("Error checking/creating index: %s", e)
        raise e

    try:
        # Obtain a handle to the index.
        index = pc.Index(index_name)
        logger.info("Connected to Pinecone index '%s'.", index_name)
        return index
    except Exception as e:
        logger.error("Error connecting to index '%s': %s", index_name, e)
        raise e

def upsert_embedding(index, celebrity_id: str, embedding: list, metadata: dict = None) -> dict:
    """
    Upsert a celebrity embedding into the Pinecone index.

    Args:
        index (Pinecone.Index): The Pinecone index object.
        celebrity_id (str): Unique identifier for the celebrity.
        embedding (list): The 512-dimensional embedding as a list of floats.
        metadata (dict): Additional metadata (e.g., name, image URL).

    Returns:
        dict: The result of the upsert operation.
    """
    vector = {
        "id": celebrity_id,
        "values": embedding,
        "metadata": metadata if metadata else {}
    }
    try:
        result = index.upsert(vectors=[vector])
        logger.info("Upserted embedding for celebrity ID %s.", celebrity_id)
        return result
    except Exception as e:
        logger.error("Error upserting embedding for celebrity ID %s: %s", celebrity_id, e)
        raise e

def query_embedding(index, query_embedding: list, top_k: int = 5, include_metadata: bool = True) -> list:
    """
    Query the Pinecone index for the most similar vectors to the query_embedding.

    Args:
        index (Pinecone.Index): The Pinecone index object.
        query_embedding (list): The query embedding vector.
        top_k (int): The number of top matches to return.
        include_metadata (bool): Whether to include metadata in the result.

    Returns:
        list: A list of matching vectors, each including id, score, and optionally metadata.
    """
    try:
        response = index.query(vector=query_embedding, top_k=top_k, include_metadata=include_metadata)
        matches = response.get("matches", [])
        logger.info("Query returned %d matches.", len(matches))
        return matches
    except Exception as e:
        logger.error("Error querying Pinecone: %s", e)
        raise e