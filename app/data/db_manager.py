"""
Module: db_manager.py

This module handles the connection to MongoDB Atlas and provides basic CRUD
functions for storing and retrieving celebrity metadata.
"""

import os
import logging
from pymongo import MongoClient, errors
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self):
        """
        Initialize the MongoDB connection.
        
        It uses the following environment variables:
            - MONGODB_URI: The MongoDB connection string.
            - MONGODB_DB_NAME: (Optional) The database name (default: 'celebrityDB').
            - MONGODB_COLLECTION: (Optional) The collection name (default: 'celebrities').
        """
        self.uri = os.getenv("MONGODB_URI")
        if not self.uri:
            logger.error("MONGODB_URI is not set in the environment variables.")
            raise ValueError("MONGODB_URI must be provided.")
        self.db_name = os.getenv("MONGODB_DB_NAME", "celebrityDB")
        self.collection_name = os.getenv("MONGODB_COLLECTION_NAME", "celebrities")
        
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info("Connected to MongoDB: Database='%s', Collection='%s'.", self.db_name, self.collection_name)
        except errors.ConnectionFailure as e:
            logger.error("Could not connect to MongoDB: %s", e)
            raise e

    def insert_document(self, document: dict):
        """
        Insert a single document into the collection.
        
        Args:
            document (dict): Celebrity metadata (e.g., id, name, biography, image_url, etc.)
        
        Returns:
            The inserted document's _id.
        """
        try:
            result = self.collection.insert_one(document)
            logger.info("Inserted document with _id: %s", result.inserted_id)
            return result.inserted_id
        except Exception as e:
            logger.error("Error inserting document: %s", e)
            raise e

    def insert_documents(self, documents, batch_size=1000):
        """
        Insert documents in batches to avoid connection timeouts.
        Args:
            documents (list): List of documents to insert.
            batch_size (int): Number of documents per batch.
        Returns:
            list: List of inserted document IDs.
        """
        inserted_ids = []
        try:
            for i in range(0, len(documents), batch_size):
                logger.info("Inserting batch %d to %d...", i, i+batch_size)
                batch = documents[i:i+batch_size]
                result = self.collection.insert_many(batch)
                inserted_ids.extend(result.inserted_ids)
            return inserted_ids
        except Exception as e:
            self.logger.error("Error inserting documents: %s", e)
            raise e
        
    def find_document(self, query: dict):
        """
        Retrieve a single document based on a query.
        
        Args:
            query (dict): MongoDB query.
        
        Returns:
            dict or None: The found document or None if not found.
        """
        try:
            document = self.collection.find_one(query)
            logger.info("Found document: %s", document)
            return document
        except Exception as e:
            logger.error("Error finding document: %s", e)
            raise e

    def update_document(self, query: dict, update: dict):
        """
        Update a document based on a query.
        
        Args:
            query (dict): The query to find the document.
            update (dict): The update operations to apply.
        
        Returns:
            int: The number of documents updated.
        """
        try:
            result = self.collection.update_one(query, update)
            logger.info("Updated %d document(s).", result.modified_count)
            return result.modified_count
        except Exception as e:
            logger.error("Error updating document: %s", e)
            raise e

    def delete_document(self, query: dict):
        """
        Delete a document based on a query.
        
        Args:
            query (dict): The query to find the document.
        
        Returns:
            int: The number of documents deleted.
        """
        try:
            result = self.collection.delete_one(query)
            logger.info("Deleted %d document(s).", result.deleted_count)
            return result.deleted_count
        except Exception as e:
            logger.error("Error deleting document: %s", e)
            raise e
        
    def delete_all_documents(self):
        """
        Delete all documents from the collection.
        
        Returns:
            int: The number of documents deleted.
        """
        try:
            result = self.collection.delete_many({})
            logger.info("Deleted %d document(s).", result.deleted_count)
            return result.deleted_count
        except Exception as e:
            logger.error("Error deleting documents: %s", e)

    def get_all_documents(self, limit=None):
        """
        Retrieve all documents from the collection.
        
        Returns:
            list: A list of all celebrity documents.
        """
        try:
            if limit:
                documents = list(self.collection.find().limit(limit))
            else:
                documents = list(self.collection.find())
            logger.info("Retrieved %d documents.", len(documents))
            return documents
        except Exception as e:
            logger.error("Error retrieving documents: %s", e)
            raise e