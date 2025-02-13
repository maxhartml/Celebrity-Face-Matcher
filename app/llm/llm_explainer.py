import os
import logging
from dotenv import load_dotenv
import torch
import clip
from PIL import Image
from huggingface_hub import InferenceClient
from app.config import HF_API_TOKEN, DEVICE, NUMBER_OF_CAPTIONS
from app.llm.candidate_response import cache_candidate_texts  # This function returns a list of candidate texts
from app import logging_config  # Ensure central logging is configured

# Load environment variables (if not already loaded in config)
load_dotenv()

logger = logging.getLogger("app.llm")

class LLMExplainer:
    """
    LLMExplainer integrates CLIP for captioning with Hugging Face's InferenceClient for text generation.
    It generates a human-readable explanation for the similarity between two images by comparing 
    the CLIP-generated captions and then using a language model to explain the similarities.
    """

    def __init__(self, text_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", clip_model_name: str = "ViT-B/32", device: str = DEVICE):
        """
        Initialize the LLMExplainer.
        
        This constructor loads configuration from the config and environment. It initializes 
        the Hugging Face InferenceClient for text generation and loads the specified CLIP model 
        and its preprocessing pipeline. It also loads a set of candidate captions for CLIP to choose from.
        
        Args:
            text_model_name (str): The Hugging Face model identifier for text generation.
            clip_model_name (str): The CLIP model identifier.
            device (str): The device on which to run the models ("cpu" or "cuda").
        
        Raises:
            ValueError: If the HF_API_TOKEN is not set.
            Exception: For any errors during model loading.
        """
        logger.debug("Initializing LLMExplainer with text_model_name='%s', clip_model_name='%s', device='%s'", text_model_name, clip_model_name, device)
        self.device = device

        # Initialize InferenceClient for text generation.
        self.api_token = HF_API_TOKEN
        if not self.api_token:
            logger.error("HF_API_TOKEN not set in config or environment.")
            raise ValueError("Please set your Hugging Face API token (HF_API_TOKEN) in your config/.env file.")
        try:
            self.client = InferenceClient(token=self.api_token)
            logger.info("Hugging Face InferenceClient initialized successfully.")
        except Exception as e:
            logger.error("Error initializing InferenceClient: %s", e, exc_info=True)
            raise e

        self.text_model_name = text_model_name
        logger.info("LLMExplainer text generation configured to use model '%s' on device '%s'.", 
                    text_model_name, device)

        # Load the CLIP model and preprocessing pipeline.
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            logger.info("CLIP model '%s' loaded successfully on device '%s'.", clip_model_name, self.device)
        except Exception as e:
            logger.error("Error loading CLIP model '%s': %s", clip_model_name, e, exc_info=True)
            raise e

        # Load candidate texts for caption selection.
        try:
            self.candidate_texts = cache_candidate_texts()
            logger.info("Loaded %d candidate texts for CLIP captioning.", len(self.candidate_texts))
        except Exception as e:
            logger.error("Error loading candidate texts: %s", e, exc_info=True)
            # Fallback to an empty list if loading fails.
            self.candidate_texts = []

    def generate_text_explanation(self, query_caption: str, match_caption: str) -> str:
        """
        Generate a textual explanation for the similarity between a query and a matched image.
        
        This function builds a prompt combining the two captions and uses the language model to generate
        an explanation.
        
        Args:
            query_caption (str): Caption for the query image.
            match_caption (str): Caption for the matched image.
        
        Returns:
            str: The generated explanation.
        
        Raises:
            Exception: Propagates any exception raised during text generation.
        """
        logger.debug("Generating text explanation for query_caption='%s' and match_caption='%s'", query_caption, match_caption)
        prompt = (
            f"Query image description: '{query_caption}'.\n"
            f"Matched image description: '{match_caption}'.\n"
            "Focus solely on their similarities. Describe in a cheeky and cool manner how these two images share strikingly similar features—almost like long-lost twins with a flair for style. "
            "Emphasize their common traits with a humorous twist that makes it both convincing and entertaining."
        )
        try:
            logger.info("Generating explanation using prompt.")
            response = self.client.text_generation(
                prompt=prompt,
                model=self.text_model_name,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2
            )
            cleaned_response = response.strip()
            # Remove the prompt from the generated text if present.
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
        
        This function opens and preprocesses the image, tokenizes the candidate texts,
        computes cosine similarity between image and text embeddings, and selects the best caption.
        
        Args:
            image_path (str): The file path to the image.
        
        Returns:
            str: The candidate caption with the highest similarity.
        
        Raises:
            Exception: If any step (opening, preprocessing, tokenizing, encoding) fails.
        """
        logger.debug("Generating CLIP caption for image_path='%s'", image_path)
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info("Image '%s' opened and converted to RGB.", image_path)
        except Exception as e:
            logger.error("Error opening image '%s': %s", image_path, e, exc_info=True)
            raise e

        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            logger.info("Image '%s' preprocessed successfully.", image_path)
        except Exception as e:
            logger.error("Error preprocessing image '%s': %s", image_path, e, exc_info=True)
            raise e

        try:
            # Tokenize candidate texts.
            text_inputs = torch.cat([clip.tokenize(text) for text in self.candidate_texts]).to(self.device)
            logger.info("Candidate texts tokenized successfully.")
        except Exception as e:
            logger.error("Error tokenizing candidate texts: %s", e, exc_info=True)
            raise e

        try:
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                logger.info("Image and text features encoded successfully.")
        except Exception as e:
            logger.error("Error encoding image/text with CLIP for '%s': %s", image_path, e, exc_info=True)
            raise e

        # Normalize features and compute cosine similarities.
        try:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Compute similarities as before
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            # Get the index along the candidate texts dimension (dim=1)
            best_idx = similarities.argmax(dim=1).item()  # best_idx is now in the range [0, N-1]
            # Now correctly index the similarity score using the row 0
            best_score = similarities[0, best_idx].item()
            best_caption = self.candidate_texts[best_idx]
            logger.info("CLIP selected caption: '%s' with score: %f for image '%s'.", best_caption, best_score, image_path)
            best_caption_and_score = {"caption": best_caption, "score": best_score}
            return best_caption_and_score
        except Exception as e:
            logger.error("Error computing similarities for image '%s': %s", image_path, e, exc_info=True)
            raise e

    def explain_similarity(self, query_image_path: str, match_image_path: str) -> str:
        """
        Generate an explanation for why the query image and the matched image are similar.
        
        This method generates captions for both images using CLIP and then uses the text model to explain
        the key visual similarities.
        
        Args:
            query_image_path (str): The file path to the query image.
            match_image_path (str): The file path to the matched image.
        
        Returns:
            str: A generated explanation of the similarity.
        """
        logger.debug("Explaining similarity between query_image_path='%s' and match_image_path='%s'", query_image_path, match_image_path)
        try:
            query_caption = self.get_clip_caption(query_image_path)
            logger.info("Generated caption for query image '%s': '%s'", query_image_path, query_caption)
        except Exception as e:
            logger.error("Failed to generate caption for query image '%s': %s", query_image_path, e, exc_info=True)
            query_caption = "No caption available"

        try:
            match_caption = self.get_clip_caption(match_image_path)
            logger.info("Generated caption for match image '%s': '%s'", match_image_path, match_caption)
        except Exception as e:
            logger.error("Failed to generate caption for match image '%s': %s", match_image_path, e, exc_info=True)
            match_caption = "No caption available"

        try:
            explanation = self.generate_text_explanation(query_caption, match_caption)
            logger.info("Generated explanation for similarity: '%s'", explanation)
            return explanation, query_caption, match_caption
        except Exception as e:
            logger.error("Failed to generate explanation: %s", e, exc_info=True)
            return "No explanation available."

# Example usage (for testing purposes):
if __name__ == "__main__":
    try:
        explainer = LLMExplainer()
        # Replace these with valid image paths on your system.
        query_image_path = "celeba_data/img_align_celeba/img_align_celeba/000006.jpg"
        match_image_path = "celeba_data/img_align_celeba/img_align_celeba/000021.jpg"
        explanation = explainer.explain_similarity(query_image_path, match_image_path)
        print("Explanation:", explanation)
    except Exception as main_e:
        logger.error("Error in LLMExplainer example usage: %s", main_e, exc_info=True)