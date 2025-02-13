# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Hugging Face settings
HF_API_TOKEN=os.getenv("HF_API_TOKEN")  # API token for Hugging Face

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # API key for Pinecone
PINECONE_REGION = os.getenv("PINECONE_REGION")  # Region for Pinecone
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")  # Cloud provider for Pinecone
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # Index name for Pinecone

# MongoDB settings
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")  # Password for MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")  # URI for MongoDB connection
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")  # Collection name in MongoDB

# Other project settings
DEVICE = "cpu"  # Device to run the computations on (e.g., "cpu" or "cuda")
TOP_K = 5  # Number of top results to retrieve
EMBEDDING_DIMENSION = 512  # Dimension of the embeddings
DATASET_ID = "jessicali9530/celeba-dataset"  # Dataset ID for the project
OUTPUT_DIR = os.path.join(os.getcwd(), "celeba_data")  # Directory to store output data
AUTO_INGEST_TOP_N = -1  # Set to -1 to ingest all records, otherwise specify the number of records to ingest