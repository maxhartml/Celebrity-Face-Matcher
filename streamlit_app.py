import os
import sys
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import logging
import logging_config  # Ensure central logging is configured
from dotenv import load_dotenv

# Import functions from your codebase
from app.query import query_engine
from app.image_processing import utils

# Load environment variables from .env
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="Celebrity Face Matcher", layout="wide")

# Sidebar: project description and pipeline overview
st.sidebar.title("About Celebrity Face Matcher")
st.sidebar.markdown(
    """
    **Celebrity Face Matcher** is an AI application that finds your celebrity look-alike using facial embeddings.

    **Pipeline Overview:**
    1. **Upload:** Upload your image.
    2. **Preprocessing:** The image is processed to detect and align the face.
    3. **Embedding Extraction:** A 512-dimensional embedding is computed using a deep learning model.
    4. **Vector Search:** Your embedding is compared to celebrity embeddings stored in Pinecone.
    5. **Results:** The top matching celebrity images and their attribute details are displayed and exported.
    
    This app is built using Streamlit for rapid prototyping.
    """
)

# Main title and description
st.title("Celebrity Face Matcher")
st.write("Upload an image to find your celebrity look-alike based on facial embeddings.")

# File uploader widget (accepts JPG, JPEG, PNG)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    st.markdown("### Processing Image")
    st.write("Your image is being processed through the pipeline to extract facial embeddings and query the Pinecone vector store for similar celebrity images.")
    
    # Query the image using your query_pipeline function.
    matches = query_engine.query_image(temp_file_path, device='cpu', top_k=5)
    
    if not matches:
        st.error("No matches found for the uploaded image.")
    else:
        st.markdown("### Match Details")
        match_rows = []
        for match in matches:
            row = {
                "image_id": match.get("id", ""),
                "score": match.get("score", "")
            }
            metadata = match.get("metadata", {})
            # Flatten metadata (excluding 'image_url' which is used to load the image)
            for key, value in metadata.items():
                if key != "image_url":
                    row[key] = value
            match_rows.append(row)
        st.dataframe(pd.DataFrame(match_rows))
        
        st.markdown("### Composite Query Result")
        results_folder = os.path.join(os.getcwd(), "images/results")
        composite_filename = f"{os.path.splitext(os.path.basename(temp_file_path))[0]}_query_results.jpg"
        try:
            composite_result_path = query_engine.compose_query_results(
                temp_file_path, matches, results_folder, composite_filename, target_height=224
            )
            composite_img = cv2.imread(composite_result_path)
            if composite_img is not None:
                # Convert BGR to RGB for display in Streamlit.
                composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
                st.image(composite_img, caption="Composite Query Result", use_container_width=True)
            else:
                st.error("Failed to load composite image.")
        except Exception as e:
            st.error(f"Error composing query results: {e}")
        
        st.markdown("### Download Results")
        # Download CSV button
        csv_filename = f"{os.path.splitext(os.path.basename(temp_file_path))[0]}_query_results.csv"
        csv_filepath = os.path.join(results_folder, csv_filename)
        try:
            query_engine.export_matches_to_csv(matches, csv_filepath)
            with open(csv_filepath, "rb") as csv_file:
                st.download_button(
                    label="Download Match Results CSV",
                    data=csv_file,
                    file_name=csv_filename,
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error exporting query results to CSV: {e}")
        
        # Download composite image button
        try:
            with open(composite_result_path, "rb") as img_file:
                st.download_button(
                    label="Download Composite Image",
                    data=img_file,
                    file_name=composite_filename,
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"Error providing download for composite image: {e}")
    
    # Optionally, remove the temporary file if desired.
    # os.remove(temp_file_path)