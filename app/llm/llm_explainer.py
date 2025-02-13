import os
import logging
from dotenv import load_dotenv
import torch
import clip
from PIL import Image
from huggingface_hub import InferenceClient
from app.config import HF_API_TOKEN, DEVICE
from app.llm.candidate_response import load_candidate_texts  # This function should return your list of candidate texts
from app import logging_config  # Ensure central logging is configured

# Load environment variables (if not already loaded in config)
load_dotenv()

logger = logging.getLogger("app.llm")

class LLMExplainer:
    def __init__(self, text_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", clip_model_name: str = "ViT-B/32", device: str = DEVICE):
        """
        Initialize the LLMExplainer which integrates CLIP for captioning and Hugging Face's InferenceClient for text generation.
        
        Configuration is loaded from your config and environment. No runtime parameters are allowed.
        
        Args:
            text_model_name (str): The Hugging Face model identifier for text generation.
            clip_model_name (str): The CLIP model identifier.
            device (str): Device on which to run the models ("cpu" or "cuda").
        """
        self.device = device

        # Initialize InferenceClient for text generation.
        self.api_token = HF_API_TOKEN
        if not self.api_token:
            logger.error("HF_API_TOKEN not set in config or environment.")
            raise ValueError("Please set your Hugging Face API token (HF_API_TOKEN) in your config/.env file.")
        self.client = InferenceClient(token=self.api_token)
   
        self.text_model_name = text_model_name
        logger.info("LLMExplainer text generation initialized with model '%s' on device '%s'.", 
                    text_model_name, device)

        # Load CLIP model and its preprocessing pipeline.
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            logger.info("CLIP model '%s' loaded on device '%s'.", clip_model_name, self.device)
        except Exception as e:
            logger.error("Error loading CLIP model '%s': %s", clip_model_name, e, exc_info=True)
            raise e

        # Load candidate texts for CLIP captioning (e.g., 200 varied candidate captions).
        try:
            self.candidate_texts = load_candidate_texts(200)
            logger.info("Loaded %d candidate texts for CLIP captioning.", len(self.candidate_texts))
        except Exception as e:
            logger.error("Error loading candidate texts: %s", e, exc_info=True)
            self.candidate_texts = []  # Fallback to empty list

    def generate_text_explanation(self, query_caption: str, match_caption: str) -> str:
        """
        Generate a textual explanation for the similarity between a query image and a matched image.
        
        Args:
            query_caption (str): Caption for the query image.
            match_caption (str): Caption for the matched image.
        
        Returns:
            str: A generated explanation.
        """
        prompt = (
            f"Query image description: '{query_caption}'.\n"
            f"Matched image description: '{match_caption}'.\n"
            "Explain the key visual similarities between these images concisely."
        )
        try:
            logger.info("Generating explanation using prompt.")
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
        Use CLIP to choose the best caption for an image from the candidate texts.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            str: The caption with the highest similarity.
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

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarities.argmax().item()
        best_caption = self.candidate_texts[best_idx]
        logger.info("CLIP selected caption: '%s' for image '%s'.", best_caption, image_path)
        return best_caption

    def explain_similarity(self, query_image_path: str, match_image_path: str) -> str:
        """
        Generate an explanation for why the query image and the matched image are similar.
        
        This method uses CLIP to generate captions for both images and then generates an explanation via the text model.
        
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

# Example usage:
if __name__ == "__main__":
    try:
        explainer = LLMExplainer(text_model_name="gpt2", clip_model_name="ViT-B/32", device="cpu")
        query_image_path = "celeba_data/img_align_celeba/img_align_celeba/000006.jpg"
        match_image_path = "celeba_data/img_align_celeba/img_align_celeba/000021.jpg"
        explanation = explainer.explain_similarity(query_image_path, match_image_path)
        print("Explanation:", explanation)
    except Exception as main_e:
        logger.error("Error in LLMExplainer example usage: %s", main_e, exc_info=True)