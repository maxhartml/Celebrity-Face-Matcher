# 🌟 Celebrity Face Matcher 🌟

Welcome to **Celebrity Face Matcher**, a cutting-edge project that empowers individuality by matching your facial image to a celebrity look-alike. Leveraging state-of-the-art deep learning models for face detection, alignment, and embedding extraction, this system not only boosts self-confidence but also paves the way for interactive, real-time storytelling experiences.

## 📚 Table of Contents
- [Overview](#overview)
- [Vision](#vision)
- [Technology Stack](#technology-stack)
- [Architecture and Pipeline](#architecture-and-pipeline)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

**Celebrity Face Matcher** is designed to demonstrate that everyone has a unique, star-quality look. It processes user-uploaded images to detect, align, and generate facial embeddings. These embeddings are then compared against a pre-populated database of celebrity embeddings (using a managed vector database—Pinecone) to find the top matching celebrities. Additionally, a natural language explanation module (planned for future iterations) will provide personalized feedback on why you match with a particular celebrity.

## 🎯 Vision

Our mission is to:
- **Empower Individuality**: Show users that every face is unique and special.
- **Boost Self-Confidence**: Help users see themselves as the stars they are.
- **Democratize Celebrity Culture**: Break traditional barriers by demonstrating that anyone can resemble a celebrity.
- **Real-Time Storytelling**: Lay the groundwork for an interactive cinematic experience where users can turn everyday life into a dynamic story.

## 🛠️ Technology Stack

- **Python 3.12+**: The primary programming language used throughout the project.
- **Deep Learning Models**:
    - **MTCNN**: For robust face detection and landmark extraction.
    - **InceptionResnetV1**: For generating 512-dimensional facial embeddings (pre-trained on VGGFace2).
- **Vector Database**:
    - **Pinecone**: A managed vector database service used to store and query high-dimensional facial embeddings using cosine similarity.
- **Database**:
    - **MongoDB Atlas**: Stores celebrity metadata and image paths. We ingest a subset of the CelebA dataset for this purpose.
- **APIs and Tools**:
    - **Kaggle API**: Used to download the CelebA dataset from Kaggle.
    - **FastAPI/Flask (Planned)**: For future deployment as a web service.
    - **PyTorch**: Used for deep learning model inference.
- **Other Libraries**:
    - **OpenCV**: For image reading, processing, and visualization.
    - **Pandas**: For CSV file processing.
    - **dotenv**: For managing environment variables.

## 🏗️ Architecture and Pipeline

The project is organized into several key components:

1. **Data Ingestion**:
     - The `app/data/celeba_ingestion.py` module downloads a subset of the CelebA dataset using the Kaggle API, processes the attribute CSV files, and stores celebrity records (including a local image path, simulated name, and age) in MongoDB.

2. **Image Processing Pipeline**:
     - The `app/image_processing/` package includes modules for:
         - **Face Detection (`face_detector.py`)**: Uses MTCNN to locate faces in an image.
         - **Face Alignment (`face_aligner.py`)**: Aligns detected faces based on facial landmarks.
         - **Embedding Extraction (`face_encoder.py`)**: Extracts a 512-dimensional embedding from an aligned face using a pre-trained model.
         - **Pipeline Orchestration (`run_pipeline.py`)**: Integrates the above steps, creates composite images for visualization, and saves outputs.

3. **Vector Storage**:
     - The `app/vector_store/pinecone_client.py` module handles:
         - **Index Initialization**: Automatically creates or connects to a Pinecone index.
         - **Upserting Embeddings**: Inserts (or updates) celebrity embeddings along with metadata.
         - **Querying**: Allows for similarity search based on a user’s query embedding.

4. **Query & Inference**:
     - The `app/query/query_pipeline.py` module takes a user image, processes it through the image pipeline to obtain an embedding, and queries the Pinecone index to retrieve the top 5 similar celebrity embeddings. It also composes a composite result image showing the query image alongside the top matches.

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.12 or later
- MongoDB Atlas account (for storing celebrity data)
- Pinecone account (for vector storage)
- Kaggle account (with API credentials)

### Steps

1. **Clone the Repository**:
     ```sh
     git clone https://github.com/maxhartml/Celebrity-Face-Matcher.git
     cd Celebrity-Face-Matcher
     ```

2. **Set Up a Virtual Environment**:
     ```sh
     python -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
     ```

3. **Install the Dependencies**:
     ```sh
     pip install -r requirements.txt
     ```

4. **Configure Environment Variables**:
     Create a `.env` file in the root of your project and add the following (replace placeholder values with your actual credentials):
     ```env
     # MongoDB
     MONGODB_URI=your_mongodb_connection_string
     MONGODB_DB_NAME=celebrityDB
     MONGODB_COLLECTION=celebrities

     # Pinecone
     PINECONE_API_KEY=your_pinecone_api_key
     PINECONE_INDEX_NAME=celebrity-embeddings
     PINECONE_CLOUD=aws
     PINECONE_REGION=us-east-1

     # Kaggle (for dataset ingestion)
     # Ensure you have your kaggle.json file in ~/.kaggle/
     CELEBA_DATASET_ID=jessicali9530/celeba-dataset
     CELEBA_OUTPUT_DIR=./celeba_data
     ```

5. **Set Up Kaggle Credentials**:
     Follow Kaggle’s instructions to place your `kaggle.json` file in `~/.kaggle/` or set the `KAGGLE_CONFIG_DIR` environment variable.

## 🚀 Usage

### Data Ingestion

To download and ingest a subset of the CelebA dataset into MongoDB, run:
```sh
python -m app.data.celeba_ingestion
```
This script will download the dataset (if not already present), parse the attribute CSV file, and insert the top 100 celebrity records into MongoDB.
### Running the Image Processing Pipeline

To process all celebrity records from the database, extract their embeddings, and store them in Pinecone, run:

```sh
python -m app.orchestration.orchestrator
```

This module retrieves records from MongoDB, downloads each celebrity image (using a local file path), processes the images through the pipeline, and upserts the embeddings (with metadata) into the Pinecone vector store.

### Querying the Pinecone Index

To query the Pinecone index with an image and retrieve the top 5 matching celebrity embeddings, run:
```sh
python -m app.query.query_pipeline /path/to/your/test_image.jpg
```
This will:
- Process the image through the pipeline.
- Query the Pinecone index using the extracted embedding.
- Compose and save a composite result image (which shows the query image alongside the top 5 matching images) in `images/results`.


## 🤝 Contributing

Contributions are welcome! Please feel free to fork this repository and submit pull requests. When contributing, please adhere to our code style guidelines and ensure your changes are well tested.

## 📜 License

This project is licensed under the MIT License.

We hope you find **Celebrity Face Matcher** inspiring and easy to work with. Enjoy exploring your unique star quality!

If you have any questions or need further assistance, please open an issue or contact us.

Happy coding! 🌟