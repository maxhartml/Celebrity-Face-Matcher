import random
import clip
from app.config import NUMBER_OF_CAPTIONS

def is_text_within_context(text: str, max_tokens: int = 77) -> bool:
    """
    Check if the given text, when tokenized by CLIP's tokenizer, has at most max_tokens.
    
    Args:
        text (str): The text to check.
        max_tokens (int): Maximum allowed tokens.
        
    Returns:
        bool: True if token count <= max_tokens, else False.
    """
    try:
        tokens = clip.tokenize(text)  # Returns a tensor of shape (1, token_count)
        token_count = tokens.shape[1]
        return token_count <= max_tokens
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return False

def load_candidate_texts() -> list:
    """
    Dynamically generate a set of concise candidate captions describing facial attributes.
    Each caption is a single sentence that lists key features, separated by commas.
    Only captions within CLIP's 77-token limit are accepted.
    
    Returns:
        list: A sorted list of candidate caption strings.
    """
    # Expanded, yet concise, descriptor lists.
    adjectives = [
        "cheerful", "confident", "mysterious", "elegant", "friendly",
        "vibrant", "thoughtful", "radiant", "serene", "intense",
        "charismatic", "engaging", "dynamic", "refined", "exuberant",
        "melancholic", "somber", "grim", "haggard", "robust",
        "delicate", "stark", "bold", "uninhibited", "subdued"
    ]
    
    eye_descriptions = [
        "bright eyes", "deep eyes", "soulful eyes", "piercing eyes", "warm eyes",
        "vivid eyes", "clear eyes", "gentle eyes", "intriguing eyes", "striking eyes",
        "dull eyes", "bleary eyes", "narrow eyes", "glazed eyes", "sharp eyes"
    ]
    
    hair_descriptions = [
        "neat hair", "wavy hair", "edgy cut", "long hair", "well-groomed hair",
        "curly hair", "slick hair", "elegant hair", "textured hair", "artful hair",
        "unkempt hair", "frizzy hair", "sparse hair", "thick hair", "short hair"
    ]
    
    facial_features = [
        "defined jawline", "sculpted cheekbones", "inviting smile", "strong chin",
        "balanced features", "delicate features", "expressive brows", "engaging face",
        "asymmetrical features", "rough features", "weathered face", "sharp features",
        "smooth features", "facial scars"
    ]
    
    skin_descriptions = [
        "clear skin", "radiant skin", "flawless complexion", "luminous skin", "glowing skin",
        "soft skin", "smooth skin", "blemished skin", "rough skin", "tanned skin",
        "pale skin", "oily skin", "scarred skin", "sallow skin", "dull skin"
    ]
    
    expression_descriptions = [
        "joyful smile", "thoughtful gaze", "calm look", "lively demeanor", "subtle smirk",
        "warm expression", "focused look", "sullen look", "vacant stare", "intense glare",
        "scowling", "content smile"
    ]
    
    age_descriptions = [
        "early 20s", "mid 20s", "late 20s", "early 30s", "mid 30s",
        "late 30s", "early 40s", "mid 40s", "late 40s", "50s",
        "youthful", "mature", "aged", "young", "old"
    ]
    
    candidate_texts = set()
    
    # Generate candidate captions until we have the desired number.
    while len(candidate_texts) < NUMBER_OF_CAPTIONS:
        caption = (
            f"{random.choice(adjectives)}, {random.choice(eye_descriptions)}, "
            f"{random.choice(hair_descriptions)}, {random.choice(facial_features)}, "
            f"{random.choice(skin_descriptions)}, {random.choice(expression_descriptions)}, "
            f"{random.choice(age_descriptions)}"
        )
        # Check that the caption is within the token limit.
        if is_text_within_context(caption):
            candidate_texts.add(caption)
    
    return sorted(list(candidate_texts))