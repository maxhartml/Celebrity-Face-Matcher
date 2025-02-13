import os
import tempfile
import streamlit as st
import cv2
import numpy as np
import app.logging_config as logging_config  # Ensure central logging is configured
from dotenv import load_dotenv

# Import the QueryEngine class from our refactored query module.
from app.query.query_engine import QueryEngine

# Load environment variables from .env
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Celebrity Face Matcher", layout="wide")

# Main Title and Description
st.title("Celebrity Face Matcher")
st.write(
    """
    **Discover Your Celebrity Twin!**  
    Upload your image to see which celebrity you most closely resemble, based on deep facial features.
    
    This application uses:
    - **MTCNN** for robust face detection and alignment  
    - **InceptionResnetV1** (trained on VGGFace2) for generating 512-dimensional face embeddings  
    - **Pinecone**, a high-performance vector database (ANN-based) for nearest-neighbor retrieval  
    - **CLIP** to dynamically create and select the best text-based “candidate captions” describing each image  
    - **A Hugging Face LLM** (e.g., Mistral-7B) to generate a human-friendly explanation of the match

    By combining these components, we can robustly identify a celebrity match and even explain *why* they're a match!
    """
)

# Pipeline Overview
st.markdown("### Pipeline Overview")
st.write(
    """
    1. **Face Detection & Alignment (MTCNN):** Locate your face using a cascaded convolutional network and align it for consistency.
    2. **Embedding Extraction (InceptionResnetV1):** Convert your aligned face into a 512-D vector capturing intricate facial features.
    3. **Vector Search (Pinecone):** Retrieve the closest match from an index of over 10,000 embeddings using ANN and cosine similarity.
    4. **Composite Image:** Visually compare your face with the top match.
    5. **LLM Explanation (CLIP + HF Inference):** Generate candidate captions from your image and the match, then use an LLM to produce an explanation.
    """
)

# File Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_filename = uploaded_file.name
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, original_filename)

    # Save the uploaded file to a temp file
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Display the user's uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=340)

    # Processing Spinner
    with st.spinner("Processing your image through the pipeline..."):
        engine = QueryEngine()
        try:
            results = engine.run_query(temp_file_path)
        except Exception as e:
            st.error(f"Error running query: {e}")
            st.stop()

    # Unpack the results
    composite_image_path = results.get("composite_image", "")
    csv_file_path = results.get("csv_file", "")
    query_embedding = results.get("query_embedding", [])
    # 'probs' is from MTCNN face detection – the confidence in detecting the face.
    face_detection_confidence = results.get("probs", None)
    processed_image_path = results.get("processed_image", "")
    matches = results.get("matches", [])
    explanation = results.get("explanation", "No explanation available.")
    query_caption = results.get("query_caption", {})
    match_caption = results.get("match_caption", {})

    # Extract top-match details
    similarity_score = matches[0]["score"] if (matches and "score" in matches[0]) else "No score available"
    match_embedding = matches[0].get("values", []) if (matches and "values" in matches[0]) else []

    # ------------------------
    # STEP 1: Face Detection & Alignment (MTCNN)
    # ------------------------
    st.markdown("## Step 1: Face Detection & Alignment (MTCNN)")
    st.write(
        f"""
        **What We Did:**  
        - **MTCNN** uses a cascaded convolutional network approach to accurately detect faces under various conditions such as changes in pose, lighting, or partial occlusion.
        - Once detected, it extracts facial landmarks (e.g., eyes, nose, mouth) and applies an affine transformation to align the face—rotating, scaling, and cropping it to a canonical view.
        
        **Face Detection Confidence:** {face_detection_confidence}
        
        **Why This Matters:**  
        Standardizing the face’s orientation and scale minimizes variability, ensuring the subsequent embedding extraction is both accurate and consistent.
        """
    )
    if os.path.exists(processed_image_path):
        processed_img = cv2.imread(processed_image_path)
        if processed_img is not None:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(processed_img, caption="Processed & Aligned Face", width=320)
        else:
            st.error("Processed image could not be loaded.")
    else:
        st.error("Processed image file not found.")

    # ------------------------
    # STEP 2: Embedding Extraction (InceptionResnetV1)
    # ------------------------
    st.markdown("## Step 2: Embedding Extraction (InceptionResnetV1)")
    st.write(
        """
        **What We Did:**  
        - We passed your aligned face into **InceptionResnetV1**, a convolutional neural network pre-trained on the extensive **VGGFace2** dataset.
        - The model outputs a **512-dimensional** embedding that encodes detailed facial features such as eye shape, jawline, and texture.

        **Why This Matters:**  
        These embeddings provide a robust, mathematical representation of your face, enabling accurate comparison even under varying conditions (e.g., different lighting or minor pose variations).
        """
    )
    st.write("**Your Facial Embedding (first 10 values):**", query_embedding[:10], "...")
    st.markdown("---")

    # ------------------------
    # STEP 3: Vector Search in Pinecone
    # ------------------------
    st.markdown("## Step 3: Vector Search (Pinecone)")
    st.write(
        f"""
        **What We Did:**  
        - Your 512-D facial embedding was queried against **Pinecone**—a specialized vector database that implements Approximate Nearest Neighbor (ANN) search.
        - Our Pinecone index contains over **10,000 embeddings**, all precomputed and inserted using a 1xA100 GPU on Lambda Labs for rapid processing.
        - Cosine similarity (computed as the dot product between L2-normalized vectors) is used, with a scaling factor (100.0) and softmax normalization applied to produce a probability distribution over candidate matches.
        
        **Similarity Score:** {similarity_score}
        
        **Why This Matters:**  
        - **ANN methods** in Pinecone allow for efficient retrieval from massive datasets.
        - A higher similarity score indicates a greater overlap in facial features between your image and the celebrity match.
        """
    )

    # Embedding Comparison (Side-by-Side)
    st.markdown("### Embedding Comparison (Query vs. Match)")
    col_query, col_match = st.columns(2)
    with col_query:
        st.subheader("Your Embedding")
        st.write(query_embedding[:10], "...")
    with col_match:
        st.subheader("Matched Celebrity Embedding")
        st.write(match_embedding[:10], "..." if match_embedding else "No match embedding available")
    st.markdown("top 10 values of the embeddings are displayed.")
    st.markdown("---")

    # ------------------------
    # STEP 4: Composite Image
    # ------------------------
    st.markdown("## Step 4: Composite Image")
    st.write(
        """
        **What We Did:**  
        - Created a composite image by juxtaposing your face with that of the best-matching celebrity.
        
        **Why This Helps:**  
        - While numerical similarity scores are valuable, a visual comparison often provides intuitive confirmation of the match.
        """
    )
    if os.path.exists(composite_image_path):
        composite_img = cv2.imread(composite_image_path)
        if composite_img is not None:
            composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
            st.image(composite_img, caption="Composite Match Result", width=440)
        else:
            st.error("Composite image could not be loaded.")
    else:
        st.error("Composite image file not found.")
    st.markdown("---")

    # ------------------------
    # STEP 5: LLM Explanation (CLIP + HF Inference)
    # ------------------------
    st.markdown("## Step 5: LLM Explanation (CLIP + Hugging Face Inference)")
    st.write(
        """
        **What We Did with CLIP:**  
        - **CLIP** is trained on millions of image-text pairs, allowing it to learn a shared embedding space for both modalities.
        - We dynamically generate multiple candidate captions describing facial attributes (e.g., “refined, intriguing eyes, well-groomed hair…”).
        - Each caption is tokenized and passed through CLIP’s text encoder to produce a text embedding.
        - These text embeddings are then compared (using cosine similarity, after L2 normalization, scaling by 100, and applying softmax) with the image’s embedding.
        - The candidate caption with the highest softmax probability is selected as the best description.

        **Why This Matters:**  
        - This dynamic, zero-shot captioning approach allows us to generate a tailored description for each image.
        - It bridges the gap between visual data and human-readable text, enabling us to feed these captions into a large language model.
        
        **Then, Hugging Face InferenceClient:**  
        - We take the best captions for both the query and matched images and feed them into a powerful language model (Mistral-7B) to generate a cohesive explanation.
        - This explanation highlights the shared attributes between your face and the celebrity match.
        """
    )
    # Display the CLIP Captions
    st.markdown("### CLIP Captions for Each Image")
    query_cap = query_caption.get('caption', 'N/A')
    query_score = query_caption.get('score', 'N/A')
    match_cap = match_caption.get('caption', 'N/A')
    match_score = match_caption.get('score', 'N/A')

    st.write(f"**Query Image Caption:** *{query_cap}*  \n(Score: {query_score})")
    st.write(f"**Matched Image Caption:** *{match_cap}*  \n(Score: {match_score})")

    st.markdown(
        """
        **Interpreting Caption Scores:**  
        - These scores (ranging from 0 to 1) reflect how well CLIP believes each caption describes the image.
        - A higher score means stronger alignment between the textual description and the visual features.
        - For example, a score of 0.036 versus 0.023 indicates that the first caption is moderately better at capturing the image's characteristics.
        
        ---
        **Final AI-Generated Explanation:**  
        """
    )
    st.write(explanation)

    # (No download buttons are provided.)