# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# MongoDB settings
MONGODB_PASSWORD= os.getenv("MmTnsBP1bIXSM7c1")
MONGODB_URI = os.getenv("mongodb+srv://maxhart5000:MmTnsBP1bIXSM7c1@celebrity-face-matcher.gb9px.mongodb.net/")
MONGO_COLLECTION_NAME = os.getenv("celebrities")

# Other project settings
DEVICE = "cpu"
TOP_K = 5
EMBEDDING_DIMENSION = 512
DATASET_ID = "jessicali9530/celeba-dataset"
OUTPUT_DIR = os.path.join(os.getcwd(), "celeba_data")
AUTO_INGEST_TOP_N = -1  # Set to -1 to ingest all records