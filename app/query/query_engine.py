import os
import csv
import cv2
import numpy as np
import logging
import app.logging_config  # Ensure central logging is configured
from app.vector_store.pinecone_client import PineconeClient
from app.image_processing import run_pipeline, utils
from app.config import DEVICE, TOP_K
from app.llm.llm_explainer import LLMExplainer

logger = logging.getLogger("app.query.query_engine")

class QueryEngine:
    def __init__(self):
        """
        Initialize the QueryEngine.
        
        Uses DEVICE and TOP_K from config.py. Creates an instance of the PineconeClient and
        an ExplanationGenerator for generating textual explanations.
        """
        self.device = DEVICE
        self.top_k = TOP_K
        self.pc_client = PineconeClient()  
        self.index = self.pc_client.index
        self.explainer = LLMExplainer()
        logger.info("QueryEngine initialized on device '%s' with index '%s'.", self.device, self.pc_client.index_name)

    def process_query_image(self, image_path: str) -> list:
        """
        Process an image from disk to extract facial embeddings.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            list: The first facial embedding as a list.
        """
        logger.info("Processing query image: %s", image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # process_image() returns a tuple (composite_image, embeddings, processed_image_path)
        embedding, probs, processed_image_path = run_pipeline.process_image(
            image_path, device=self.device, name=base_name
        )
        logger.info("Extracted %d embeddings from image '%s'.", len(embedding), image_path)
        
        # Use an explicit check to avoid ambiguous boolean evaluation:
        if embedding is None or (hasattr(embedding, "size") and embedding.size == 0):
            logger.error("No embeddings extracted from image %s", image_path)
            return []
            
        query_vector = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        logger.info("Extracted query embedding for image '%s'.", image_path)
        return query_vector, probs, processed_image_path

    def query_index(self, query_vector: list) -> list:
        """
        Query the Pinecone index for the top_k matches.
        
        Args:
            query_vector (list): The facial embedding to query.
        
        Returns:
            list: A list of match dictionaries.
        """
        try:
            matches = self.pc_client.query_embedding(query_vector, include_metadata=True)
            logger.info("Query returned %d matches.", len(matches))
            return matches
        except Exception as e:
            logger.error("Error querying Pinecone index: %s", e, exc_info=True)
            return []

    def compose_results_image(self, query_image_path: str, matches: list, target_height: int = 224) -> str:
        """
        Compose a composite image that includes the query image and its top matching images.
        
        Args:
            query_image_path (str): Path to the query image.
            matches (list): A list of match dictionaries.
            target_height (int): The desired height for resizing images.
        
        Returns:
            str: The full path to the saved composite image.
        """
        query_img = cv2.imread(query_image_path)
        if query_img is None:
            raise ValueError(f"Could not load query image from {query_image_path}")
        q_h, q_w = query_img.shape[:2]
        scale = target_height / q_h
        query_resized = cv2.resize(query_img, (int(q_w * scale), target_height))
        
        match_images = []
        for match in matches:
            metadata = match.get("metadata", {})
            match_img_path = metadata.get("image_url")
            if not match_img_path:
                logger.warning("No image path in metadata for match ID %s.", match.get("id"))
                continue
            # Adjust image path if necessary.
            match_img_path = utils.adjust_image_path(match_img_path)
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
        results_folder = os.path.join(os.getcwd(), "images", "results")
        composite_filename = f"{os.path.splitext(os.path.basename(query_image_path))[0]}_query_results.jpg"
        saved_path = utils.save_image(composite_image, results_folder, composite_filename)
        logger.info("Composite query result image saved to %s", saved_path)
        return saved_path

    def export_matches_to_csv(self, matches: list, csv_filepath: str) -> str:
        """
        Export query match results to a CSV file.
        
        Args:
            matches (list): A list of match dictionaries from Pinecone.
            csv_filepath (str): Full path to the CSV file to be created.
        
        Returns:
            str: The CSV filepath.
        """
        attribute_keys = set()
        for match in matches:
            metadata = match.get("metadata", {})
            for key in metadata.keys():
                if key != "image_url":
                    attribute_keys.add(key)
        attribute_keys = sorted(attribute_keys)
        headers = ["image_id", "score"] + attribute_keys
        rows = []
        totals = {key: 0 for key in attribute_keys}
        for match in matches:
            row = {"image_id": match.get("id", ""), "score": match.get("score", "")}
            metadata = match.get("metadata", {})
            for key in attribute_keys:
                value = metadata.get(key, 0)
                row[key] = value
                try:
                    totals[key] += float(value)
                except Exception:
                    totals[key] += 0
            rows.append(row)
        try:
            with open(csv_filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                totals_row = {"image_id": "Total", "score": ""}
                totals_row.update(totals)
                writer.writerow(totals_row)
            logger.info("Exported query results to CSV file: %s", csv_filepath)
        except Exception as e:
            logger.error("Error exporting query results to CSV: %s", e, exc_info=True)
            raise e
        return csv_filepath

    def run_query(self, query_image_path: str) -> dict:
        """
        Run the complete query pipeline:
        - Process the query image.
        - Query the Pinecone index.
        - Compose composite results image.
        - Export match results to CSV.
        - Generate a textual explanation using the LLM module.
        
        Args:
            image_path (str): Path to the query image.
        
        Returns:
            dict: A dictionary with keys 'composite_image', 'csv_file', 'matches', and 'explanation'.
        """
        query_vector, probs, processed_image_path = self.process_query_image(query_image_path)
        if not query_vector:
            raise ValueError("Failed to extract a valid embedding from the query image.")
        matches = self.query_index(query_vector)
        if not matches:
            raise ValueError("No matches found for the query image.")
        composite_image_path = self.compose_results_image(query_image_path, matches)
        csv_filename = f"{os.path.splitext(os.path.basename(query_image_path))[0]}_query_results.csv"
        csv_filepath = os.path.join(os.getcwd(), "images", "results", csv_filename)
        self.export_matches_to_csv(matches, csv_filepath)
        # For explanation generation, use the top match only.
        top_match = matches[0]
        match_img_path = top_match.get("metadata", {}).get("image_url")
        if not match_img_path:
            explanation = "No explanation available (no match image URL)."
        else:
            # Adjust path if necessary.
            match_img_path = utils.adjust_image_path(match_img_path)
            # Use the new LLM module for explanation.
            explanation, query_caption, match_caption = self.explainer.explain_similarity(query_image_path, match_img_path)
            
            package_up = {
                "composite_image": composite_image_path,
                "csv_file": csv_filepath,
                "query_embedding": query_vector,
                "probs": probs,
                "processed_image": processed_image_path,
                "matches": matches,
                "explanation": explanation,
                "query_caption": query_caption,
                "match_caption": match_caption
            }
                
        return package_up
