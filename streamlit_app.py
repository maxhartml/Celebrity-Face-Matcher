import os
import tempfile
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import logging
import app.logging_config as logging_config  # Ensure central logging is configured
from dotenv import load_dotenv

# Import the QueryEngine class from our refactored query module.
from app.query.query_engine import QueryEngine

# Load environment variables from .env
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Celebrity Face Matcher", layout="wide")

# Sidebar: project description and pipeline overview
st.sidebar.title("About Celebrity Face Matcher")
st.sidebar.markdown(
    """
    **Celebrity Face Matcher** is an AI application that finds your celebrity look-alike using facial embeddings.

    **Pipeline Overview:**
    1. **Upload:** Upload your image.
    2. **Processing:** The image is processed to detect and align the face and extract a 512-dimensional embedding.
    3. **Query:** Your embedding is compared to celebrity embeddings stored in Pinecone.
    4. **Results:** The top matches (with their attribute details) are returned as a composite image and exported to CSV, along with a generated explanation.
    """
)

# Main title and description
st.title("Celebrity Face Matcher")
st.write("Upload an image to find your celebrity look-alike based on facial embeddings.")

# File uploader widget (accepts JPG, JPEG, PNG)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Use the original filename from the uploaded file.
    original_filename = uploaded_file.name  # e.g. "Dylan.jpg"
    # Save the uploaded file to a temporary directory with its original name.
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, original_filename)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.image(uploaded_file, caption="Uploaded Image", width=300)
    st.markdown("### Processing and Querying")
    st.write("Your image is being processed through the pipeline and queried against celebrity embeddings...")

    # Instantiate the QueryEngine (which uses your hardcoded config in your codebase)
    engine = QueryEngine()

    # Run the complete query pipeline (this call will save the composite image and CSV only once)
    try:
        results = engine.run_query(temp_file_path)
    except Exception as e:
        st.error(f"Error running query: {e}")
        st.stop()

    matches = results.get("matches", [])
    composite_image_path = results.get("composite_image", "")
    csv_file_path = results.get("csv_file", "")
    explanation = results.get("explanation", "No explanation available.")

    if not matches:
        st.error("No matches found for the uploaded image.")
    else:
        st.markdown("### Match Details")
        # Build a DataFrame from the match results.
        rows = []
        for match in matches:
            row = {"image_id": match.get("id", ""), "score": match.get("score", "")}
            metadata = match.get("metadata", {})
            # Flatten metadata (excluding 'image_url' which is used for loading the image)
            for key, value in metadata.items():
                if key != "image_url":
                    row[key] = value
            rows.append(row)
        st.dataframe(pd.DataFrame(rows))

        st.markdown("### Composite Query Result")
        composite_img = cv2.imread(composite_image_path)
        if composite_img is not None:
            # Convert from BGR to RGB for Streamlit display.
            composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
            st.image(composite_img, caption="Composite Query Result", use_container_width=True)
        else:
            st.error("Failed to load composite image.")

        st.markdown("### Explanation")
        st.write(explanation)

        st.markdown("### Download Results")
        # Download CSV button
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "rb") as csv_file:
                st.download_button(
                    label="Download Match Results CSV",
                    data=csv_file,
                    file_name=os.path.basename(csv_file_path),
                    mime="text/csv"
                )
        else:
            st.error("CSV file not found.")
        
        # Download composite image button
        if os.path.exists(composite_image_path):
            with open(composite_image_path, "rb") as img_file:
                st.download_button(
                    label="Download Composite Image",
                    data=img_file,
                    file_name=os.path.basename(composite_image_path),
                    mime="image/jpeg"
                )
        else:
            st.error("Composite image file not found.")