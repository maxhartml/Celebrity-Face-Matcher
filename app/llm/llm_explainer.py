import os
import logging
from dotenv import load_dotenv
import clip
import torch
from PIL import Image
from huggingface_hub import InferenceClient
from app.llm.candiate_response import load_candidate_texts

from app.config import HF_API_TOKEN, DEVICE 

logger = logging.getLogger("app.llm.language_model")

class LLMExplainer:
    def __init__(self, text_model_name: str = "gpt2", clip_model_name: str = "ViT-B/32", device: str = DEVICE):
        """
        Initialize the LLMExplainer which integrates CLIP for captioning and Hugging Face's InferenceClient for text generation.
        
        Args:
            text_model_name (str): Hugging Face model identifier for text generation.
            clip_model_name (str): CLIP model identifier.
            device (str): Device to run models on ("cpu" or "cuda").
        """
        self.device = device
        
        # Initialize Hugging Face InferenceClient for text generation.
        self.api_token = HF_API_TOKEN
        if not self.api_token:
            logger.error("HF_API_TOKEN not set in config or environment.")
            raise ValueError("Please set your Hugging Face API token (HF_API_TOKEN) in your config/.env file.")
        
        self.client = InferenceClient(token=self.api_token)
        self.text_model_name = text_model_name
        logger.info("LLMExplainer text generation initialized with model '%s' on device '%s'.", text_model_name, device)
        
        # Load CLIP model and its preprocessing pipeline.
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            logger.info("CLIP model '%s' loaded on device '%s'.", clip_model_name, self.device)
        except Exception as e:
            logger.error("Error loading CLIP model '%s': %s", clip_model_name, e, exc_info=True)
            raise e
        
        # Load candidate texts for CLIP captioning.
        try:
            self.candidate_texts = load_candidate_texts(200)
            logger.info("Loaded %d candidate texts for CLIP captioning.", len(self.candidate_texts))
        except Exception as e:
            logger.error("Error loading candidate texts: %s", e, exc_info=True)


    def generate_text_explanation(self, query_caption: str, match_caption: str) -> str:
        """
        Generate a textual explanation for the similarity between a query and a matched image.
        
        Args:
            query_caption (str): Caption for the query image.
            match_caption (str): Caption for the matched image.
        
        Returns:
            str: A concise explanation of the visual similarities.
        """
        prompt = (
            f"Query image description: '{query_caption}'.\n"
            f"Matched image description: '{match_caption}'.\n"
            "Explain the key visual similarities between these images concisely."
        )
        try:
            logger.info("Generating explanation for the prompt.")
            response = self.client.text_generation(
                prompt=prompt,
                model=self.text_model_name,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2
            )
            cleaned_response = response.strip()
            if cleaned_response.startswith(prompt):
                explanation = cleaned_response[len(prompt):].strip()
            else:
                explanation = cleaned_response
            logger.info("Generated explanation: '%s'.", explanation)
            return explanation
        except Exception as e:
            logger.error("Error generating explanation: %s", e, exc_info=True)
            raise e
        
    def get_clip_caption(self, image_path: str) -> str:
        """
        Use CLIP to select the best caption for an image from a list of candidate texts.
        
        Args:
            image_path (str): Path to the image file.
            candidate_texts (list): List of candidate captions.
        
        Returns:
            str: The candidate caption with the highest similarity.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error("Error opening image '%s': %s", image_path, e, exc_info=True)
            raise e

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error("Error preprocessing image '%s': %s", image_path, e, exc_info=True)
            raise e

        try:
            # Tokenize all candidate texts.
            text_inputs = torch.cat([clip.tokenize(text) for text in self.candidate_texts]).to(self.device)
        except Exception as e:
            logger.error("Error tokenizing candidate texts: %s", e, exc_info=True)
            raise e

        with torch.no_grad():
            try:
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
            except Exception as e:
                logger.error("Error encoding with CLIP for image '%s': %s", image_path, e, exc_info=True)
                raise e

        # Normalize and compute cosine similarity.
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarities.argmax().item()
        best_caption = self.candidate_texts[best_idx]
        logger.info("CLIP selected caption: '%s' for image '%s'.", best_caption, image_path)
        return best_caption

    def explain_similarity(self, query_image_path: str, match_image_path: str) -> str:
        """
        Generate an explanation for why a query image matches a given image.
        
        This method uses CLIP to generate captions for both images and then generates an explanation using the text model.
        
        Args:
            query_image_path (str): Path to the query image.
            match_image_path (str): Path to the matched image.
        
        Returns:
            str: The generated explanation.
        """
        try:
            query_caption = self.get_clip_caption(query_image_path)
        except Exception as e:
            logger.error("Failed to generate caption for query image '%s': %s", query_image_path, e, exc_info=True)
            query_caption = "No caption available"

        try:
            match_caption = self.get_clip_caption(match_image_path)
        except Exception as e:
            logger.error("Failed to generate caption for match image '%s': %s", match_image_path, e, exc_info=True)
            match_caption = "No caption available"

        try:
            explanation = self.generate_text_explanation(query_caption, match_caption)
            return explanation
        except Exception as e:
            logger.error("Failed to generate explanation: %s", e, exc_info=True)
            return "No explanation available."

# Example usage (for testing purposes):
if __name__ == "__main__":
    explainer = LLMExplainer(text_model_name="gpt2", clip_model_name="ViT-B/32", device="cpu")
    # Replace these paths with valid paths on your system for testing.
    query_image_path = "celeba_data/img_align_celeba/img_align_celeba/000006.jpg"
    match_image_path = "celeba_data/img_align_celeba/img_align_celeba/000021.jpg"
    explanation = explainer.explain_similarity(query_image_path, match_image_path)
    print("Explanation:", explanation)