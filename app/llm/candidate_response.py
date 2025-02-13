import os
import json
import random
import clip
from app.config import NUMBER_OF_CAPTIONS, CACHE_FILE
import app.logging_config
import logging

logger = logging.getLogger("app.llm.candidate_response")

def is_text_within_context(text: str, max_tokens: int = 77) -> bool:
    try:
        tokens = clip.tokenize(text)  # Returns a tensor of shape (1, token_count)
        token_count = tokens.shape[1]
        return token_count <= max_tokens
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return False

def load_candidate_texts() -> list:
    try:
        logger.info("Loading candidate texts...")
        adjectives = [
            "cheerful", "confident", "mysterious", "elegant", "friendly",
            "vibrant", "thoughtful", "radiant", "serene", "intense",
            "charismatic", "engaging", "dynamic", "refined", "exuberant",
            "melancholic", "somber", "grim", "haggard", "robust",
            "delicate", "stark", "bold", "uninhibited", "subdued",
            "astonishing", "breathtaking", "majestic", "spirited", "impressive",
            "luminous", "vivid", "alluring", "mesmerizing", "dignified",
            "not attractive", "unappealing", "disheveled"
        ]
        
        eye_descriptions = [
            "bright eyes", "deep eyes", "soulful eyes", "piercing eyes", "warm eyes",
            "vivid eyes", "clear eyes", "gentle eyes", "intriguing eyes", "striking eyes",
            "dull eyes", "bleary eyes", "narrow eyes", "glazed eyes", "sharp eyes",
            "heavy-lidded eyes", "sparkling eyes", "glittering eyes", "smoldering eyes", "fiery eyes",
            "mesmerizing eyes", "crystal-clear eyes", "expressive eyes", "intense gaze"
        ]
        
        hair_descriptions = [
            "neat hair", "wavy hair", "edgy cut", "long hair", "well-groomed hair",
            "curly hair", "slick hair", "elegant hair", "textured hair", "artful hair",
            "unkempt hair", "frizzy hair", "sparse hair", "thick hair", "short hair",
            "greasy hair", "silky hair", "lustrous hair", "voluminous hair", "messy hair",
            "shaggy hair", "spiky hair", "dreadlocks", "colorful hair", "platinum hair",
            "natural hair", "styled hair", "flowing hair"
        ]
        
        facial_features = [
            "defined jawline", "sculpted cheekbones", "inviting smile", "strong chin",
            "balanced features", "delicate features", "expressive brows", "engaging face",
            "asymmetrical features", "rough features", "weathered face", "sharp features",
            "smooth features", "facial scars", "prominent nose", "small nose", "high cheekbones",
            "protruding cheekbones", "angular features", "refined nose", "soft features", "delicate bone structure",
            "chiseled features", "sturdy features", "well-defined features"
        ]
        
        skin_descriptions = [
            "clear skin", "radiant skin", "flawless complexion", "luminous skin", "glowing skin",
            "soft skin", "smooth skin", "blemished skin", "rough skin", "tanned skin",
            "pale skin", "oily skin", "scarred skin", "sallow skin", "dull skin", "freckled skin",
            "even-toned skin", "vibrant skin", "radiant complexion", "sun-kissed skin",
            "porcelain skin", "dewy skin", "aged skin"
        ]
        
        expression_descriptions = [
            "joyful smile", "thoughtful gaze", "calm look", "lively demeanor", "subtle smirk",
            "warm expression", "focused look", "sullen look", "vacant stare", "intense glare",
            "scowling", "content smile", "frowning", "serene expression", "smiling broadly",
            "mischievous grin", "pensive look", "animated expression", "mournful expression",
            "bright smile", "stoic expression", "playful expression"
        ]
        
        age_descriptions = [
            "early 20s", "mid 20s", "late 20s", "early 30s", "mid 30s",
            "late 30s", "early 40s", "mid 40s", "late 40s", "50s",
            "youthful", "mature", "aged", "young", "old", "in their prime", "seasoned",
            "old-fashioned", "vintage", "timeless"
        ]
        
        candidate_texts = set()
        
        while len(candidate_texts) < NUMBER_OF_CAPTIONS:
            caption = (
                f"{random.choice(adjectives)}, {random.choice(eye_descriptions)}, "
                f"{random.choice(hair_descriptions)}, {random.choice(facial_features)}, "
                f"{random.choice(skin_descriptions)}, {random.choice(expression_descriptions)}, "
                f"{random.choice(age_descriptions)}"
            )
            if is_text_within_context(caption):
                candidate_texts.add(caption)
        
        logger.info(f"Generated {len(candidate_texts)} candidate texts.")
        return sorted(list(candidate_texts))
    except Exception as e:
        logger.error(f"Error loading candidate texts: {e}")
        return []

def cache_candidate_texts(cache_file: str = CACHE_FILE) -> list:
    try:
        if os.path.exists(cache_file):
            logger.info(f"Loading candidate captions from cache: {cache_file}")
            with open(cache_file, "r") as f:
                candidate_texts = json.load(f)
            logger.info(f"Loaded {len(candidate_texts)} candidate captions from cache.")
            return candidate_texts
        else:
            logger.info(f"Generating candidate captions and caching to: {cache_file}")
            candidate_texts = load_candidate_texts()
            with open(cache_file, "w") as f:
                json.dump(candidate_texts, f)
            logger.info(f"Generated and cached {len(candidate_texts)} candidate captions.")
            return candidate_texts
    except Exception as e:
        logger.error(f"Error caching candidate texts: {e}")
        return []

# Example usage:
if __name__ == "__main__":
    captions = cache_candidate_texts()
    logger.info(f"Captions: {captions}")
