<div align="center">
  
# 🌟 Celebrity Face Matcher 🌟

### _Where AI Meets Your Inner Star_ ✨

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-000000?logo=pinecone&logoColor=white)](https://www.pinecone.io/)

---

</div>

Welcome to **Celebrity Face Matcher** 🎭, a revolutionary project fusing cutting-edge AI with star-powered recognition. Using state-of-the-art deep learning models, this system transforms your selfies into celebrity connections!

<details>
<summary>📚 Table of Contents</summary>

- [Overview](#-overview)
- [Vision](#-vision)
- [Technology Stack](#-technology-stack)
- [Architecture and Pipeline](#-architecture-and-pipeline)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)
- [Streamlit Interface](#streamlit-interface)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

</details>

## 🌟 Overview

**Celebrity Face Matcher** revolutionizes the way we connect with celebrity culture. Through advanced AI algorithms and vector search technology, we create meaningful connections between everyday faces and their celebrity counterparts.

## 🎯 Vision

> _"In every face lies a story waiting to be told through the lens of stardom."_

Our mission is to:
- **Empower Individuality**: Show users that every face is unique and special.
- **Boost Self-Confidence**: Help users see themselves as the stars they are.
- **Democratize Celebrity Culture**: Break traditional barriers by demonstrating that anyone can resemble a celebrity.
- **Enable Real-Time Storytelling**: Transform everyday images into interactive, cinematic narratives.
- **Explain the Match**: Provide in-depth, technical explanations of why the system believes a particular celebrity is a match.

## 🛠️ Technology Stack

- **Python 3.12+**: The primary programming language.
- **Deep Learning Models**:
     - **MTCNN**: For robust face detection and landmark extraction.
     - **InceptionResnetV1**: A CNN trained on **VGGFace2** that generates a 512-dimensional facial embedding.
     - **CLIP**: Trained on millions of image-text pairs, this model learns a shared embedding space.
- **Vector Database**:
     - **Pinecone**: A managed vector database that stores high-dimensional embeddings.
- **Database**:
     - **MongoDB Atlas**: Stores celebrity metadata and image paths from the CelebA dataset.
- **APIs and Tools**:
     - **Kaggle API**: For downloading the CelebA dataset.
     - **PyTorch**: For model inference.
     - **Hugging Face InferenceClient**: Powers our LLM (e.g., Mistral-7B) which generates natural language explanations.
- **Other Libraries**:
     - **OpenCV**: For image processing and visualization.
     - **Pandas**: For CSV processing.
     - **dotenv**: For environment variable management.

## 🏗️ Architecture and Pipeline

The project is organized into several key components:

1. **Data Ingestion**:
      - The `app/data/celeba_ingestion.py` module downloads a subset of the CelebA dataset using the Kaggle API, processes the CSV files, and stores celebrity records in MongoDB.

2. **Image Processing Pipeline**:
      - **Face Detection & Alignment (MTCNN)**: Detects faces and extracts facial landmarks, then aligns the face into a canonical view.
      - **Embedding Extraction (InceptionResnetV1)**: Generates a robust 512-dimensional embedding capturing facial features.
      - **Pipeline Orchestration**: Integrates detection, alignment, and encoding, generating composite images that juxtapose the query image with its best match.

3. **Vector Storage & Retrieval**:
      - **Pinecone Vector Database**: Manages the index containing over 10,000 precomputed celebrity embeddings, using ANN search with cosine similarity to retrieve the most similar embeddings.

4. **Query & Inference**:
      - **Query Pipeline**: Processes a user-uploaded image through the entire pipeline and queries Pinecone to retrieve top matches.
      - **LLM Explanation Module**: Uses **CLIP** to generate captions and a Hugging Face LLM to generate a detailed explanation of the match.

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.12 or later
- MongoDB Atlas account
- Pinecone account
- Kaggle account

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
      Create a `.env` file in the project root with the following:
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

      # HuggingFace
      HF_API_TOKEN=your_hf_api_totken
      ```

5. **Set Up Kaggle Credentials**:
      Follow Kaggle’s instructions to place your `kaggle.json` file in `~/.kaggle/` or set the `KAGGLE_CONFIG_DIR` environment variable.

## 🚀 Usage

### Data Ingestion

To download and ingest a subset of the CelebA dataset into MongoDB, run:
```sh
python -m app.data.celeba_ingestion
```

### Running the Image Processing Pipeline

To process all celebrity records from the database, extract their embeddings, and store them in Pinecone, run:
```sh
python -m app.orchestration.orchestrator
```

### Querying the Pinecone Index & Running the Streamlit UI

To query the Pinecone index with an image and view the results via a user-friendly interface:

1. Launch the Streamlit UI:

     ```sh
     streamlit run streamlit_app.py
     ```


## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit pull requests. Adhere to the code style guidelines and ensure your changes are well tested.

## 📜 License

This project is licensed under the MIT License.

We hope you find Celebrity Face Matcher both inspiring and technically intriguing. Enjoy exploring your unique star quality and the sophisticated AI powering your match!

If you have any questions or need further assistance, please open an issue or contact us.

---
<br>
Happy coding! 🌟
