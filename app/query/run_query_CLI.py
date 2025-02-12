import sys
import logging
import logging_config  # Ensure central logging is configured
from app.query.query_engine import QueryEngine

logger = logging.getLogger("app.query.query_engine")

def main():
    """
    CLI entry point for the query pipeline.
    Must be called with an image path as a command-line argument.
    
    Example:
      python -m app.query.query_pipeline /path/to/your/test_image.jpg
    """
    if len(sys.argv) < 2:
        logger.error("No image path provided. Please run:\npython -m app.query.query_pipeline /path/to/your/test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    engine = QueryEngine(device='cpu', top_k=5)
    try:
        results = engine.run_query(image_path)
        logger.info("Composite image saved at: %s", results["composite_image"])
        logger.info("CSV file saved at: %s", results["csv_file"])
        logger.info("Top matches:")
        for match in results["matches"]:
            logger.info("ID: %s, Score: %.4f, Metadata: %s", match.get("id"), match.get("score"), match.get("metadata"))
    except Exception as e:
        logger.error("Error running query pipeline: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()