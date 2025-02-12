import os
import logging
from pinecone import Pinecone, ServerlessSpec  # New Pinecone client API
from dotenv import load_dotenv
from app.config import *

logger = logging.getLogger(__name__)

class PineconeClient:
    def __init__(self):
        """
        Initialize the Pinecone client and connect to (or create) an index.
        All configuration is loaded from the config file and environment variables.
        """
        # Read configuration from the config module / environment.
        self.api_key = PINECONE_API_KEY
        if not self.api_key:
            logger.error("Pinecone API key not set in environment variables or config.")
            raise ValueError("Pinecone API key must be provided.")

        self.index_name = PINECONE_INDEX_NAME
        if not self.index_name:
            logger.error("Pinecone index name not set in environment variables or config.")
            raise ValueError("Pinecone index name must be provided.")

        self.dimension = EMBEDDING_DIMENSION
        self.cloud = PINECONE_CLOUD
        self.region = PINECONE_REGION

        # Create an instance of the Pinecone client.
        self.pc = Pinecone(api_key=self.api_key)
        # Create a ServerlessSpec using the cloud and region.
        self.spec = ServerlessSpec(cloud=self.cloud, region=self.region)
        # Initialize (or create) the index.
        self.index = self.initialize_index()

    def initialize_index(self):
        """
        Connect to or create the specified Pinecone index.

        Args:
            index_name (str): The name of the index.
            dimension (int): The dimensionality of the vectors.

        Returns:
            Pinecone.Index: The connected index.
        """
        try:
            existing_indexes = self.pc.list_indexes()
            if self.index_name not in existing_indexes:
                logger.info("Index '%s' not found. Creating new index with dimension %d.", self.index_name, self.dimension)
                try:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=self.spec
                    )
                except Exception as create_e:
                    if "ALREADY_EXISTS" in str(create_e):
                        logger.info("Index '%s' already exists (detected during creation).", self.index_name)
                    else:
                        logger.error("Error creating index '%s': %s", self.index_name, create_e)
                        raise create_e
            else:
                logger.info("Index '%s' already exists.", self.index_name)
        except Exception as e:
            logger.error("Error checking/creating index: %s", e)
            raise e

        try:
            index = self.pc.Index(self.index_name)
            logger.info("Connected to Pinecone index '%s'.", self.index_name)
            return index
        except Exception as e:
            logger.error("Error connecting to index '%s': %s", self.index_name, e)
            raise e

    def upsert_embedding(self, celebrity_id: str, embedding: list, metadata: dict = None) -> dict:
        """
        Upsert a celebrity embedding into the Pinecone index.

        Args:
            celebrity_id (str): Unique identifier for the celebrity.
            embedding (list): The 512-dimensional embedding as a list of floats.
            metadata (dict): Additional metadata (e.g., attributes or image URL).

        Returns:
            dict: The result of the upsert operation.
        """
        vector = {
            "id": celebrity_id,
            "values": embedding,
            "metadata": metadata if metadata is not None else {}
        }
        try:
            result = self.index.upsert(vectors=[vector])
            logger.info("Upserted embedding for celebrity ID %s.", celebrity_id)
            return result
        except Exception as e:
            logger.error("Error upserting embedding for celebrity ID %s: %s", celebrity_id, e)
            raise e

    def query_embedding(self, query_embedding: list, include_metadata: bool = True) -> list:
        """
        Query the Pinecone index for the most similar vectors to the query_embedding.

        Args:
            query_embedding (list): The query embedding vector.
            top_k (int): The number of top matches to return.
            include_metadata (bool): Whether to include metadata in the result.

        Returns:
            list: A list of matching vectors, each including id, score, and optionally metadata.
        """
        try:
            response = self.index.query(vector=query_embedding, top_k=TOP_K, include_metadata=include_metadata)
            matches = response.get("matches", [])
            logger.info("Query returned %d matches.", len(matches))
            return matches
        except Exception as e:
            logger.error("Error querying Pinecone: %s", e)
            raise e