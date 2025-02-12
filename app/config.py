import os 

DEVICE = "cpu"
TOP_K = 5
EMBEDDING_SIZE = 512

DATASET_ID = "jessicali9530/celeba-dataset"
OUTPUT_DIR = os.path.join(os.getcwd(), "celeba_data")
AUTO_INGEST_TOP_N = -1 # Set to -1 to ingest all records