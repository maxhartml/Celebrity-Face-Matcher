import random

def load_candidate_texts(num_candidates: int = 200) -> list:
    """
    Dynamically generate a set of candidate captions describing facial attributes in detail.
    
    Args:
        num_candidates (int): The number of candidate captions to generate.
        
    Returns:
        list: A sorted list of candidate caption strings that are less than 77 tokens.
    """
    # Basic adjectives and descriptions
    adjectives = [
        "cheerful", "confident", "mysterious", "elegant", "friendly",
        "vibrant", "thoughtful", "radiant", "serene", "intense",
        "charismatic", "engaging", "dynamic", "refined", "exuberant"
    ]
    
    # Detailed descriptors for eyes, hair, facial features, and skin.
    eye_descriptions = [
        "bright, sparkling eyes that captivate", "deep, soulful eyes reflecting emotion",
        "expressive eyes that reveal hidden stories", "piercing eyes that command attention",
        "warm, inviting eyes that exude kindness", "vivid eyes with an intense gaze",
        "clear, focused eyes radiating determination", "gentle eyes that comfort the soul",
        "intriguing eyes hinting at untold mysteries", "striking eyes blending boldness and softness"
    ]
    hair_descriptions = [
        "neatly styled hair framing the face perfectly", "luxurious wavy hair that adds softness",
        "a modern, edgy haircut exuding confidence", "long, flowing hair with natural elegance",
        "well-groomed hair speaking of sophistication", "distinctive curly hair bursting with character",
        "slicked-back hair emphasizing strong features", "elegantly styled hair with timeless appeal",
        "naturally textured hair that highlights uniqueness", "artfully arranged hair merging style with ease"
    ]
    facial_features = [
        "a well-defined jawline conveying strength", "sculpted cheekbones that catch the light",
        "a subtle yet inviting smile that brightens the face", "a strong chin adding character",
        "balanced features with an air of refinement", "delicate features exuding grace",
        "expressive eyebrows framing the eyes beautifully", "a captivating smile drawing you in",
        "a poised expression that exudes calm", "an engaging visage marked by striking details"
    ]
    skin_descriptions = [
        "clear, smooth skin with a natural glow", "radiant skin exuding health",
        "a flawless complexion with even tones", "luminous skin with an almost ethereal quality",
        "skin glowing with healthy radiance", "soft, natural skin that invites touch",
        "visibly smooth skin hinting at youthfulness", "skin that appears well-cared-for and vibrant",
        "an effortlessly glowing complexion", "skin with a subtle, enchanting allure"
    ]
    accessory_descriptions = [
        "sporting stylish glasses", "wearing a chic hat that complements the look",
        "adorned with subtle jewelry", "accented with tasteful earrings",
        "with a signature accessory that stands out", "showing off a trendy scarf",
        "featuring a fashionable watch", "dressed with an eye-catching necklace"
    ]
    expression_descriptions = [
        "with a beaming, joyful smile", "with a thoughtful and introspective gaze",
        "radiating a calm and composed expression", "with an animated and lively demeanor",
        "exhibiting a subtle smirk that hints at mischief", "with an earnest, engaging look",
        "displaying a warm and approachable expression", "with a look of determined focus"
    ]
    background_descriptions = [
        "set against a minimalist urban backdrop", "with a soft, blurred natural background",
        "against an artistic, abstract setting", "with a vivid and colorful backdrop",
        "set in a cozy, indoor environment", "against a serene and calming landscape",
        "with a dynamic and modern background", "against a backdrop that enhances the subject"
    ]
    
    candidate_texts = set()
    
    # Generate candidate captions until we have the desired number.
    while len(candidate_texts) < num_candidates:
        caption = (
            f"A portrait of a {random.choice(adjectives)} individual with {random.choice(eye_descriptions)}, "
            f"{random.choice(hair_descriptions)}, and {random.choice(facial_features)}. Their skin is {random.choice(skin_descriptions)}. "
            f"They are {random.choice(expression_descriptions)} and {random.choice(accessory_descriptions)}. "
            f"The image is set {random.choice(background_descriptions)}."
        )
        # Check if the caption is less than 77 tokens (approximation using whitespace splitting).
        if len(caption.split()) < 52:
            print(len(caption.split()))
            candidate_texts.add(caption)
        # Else, skip and generate a new caption.
    
    return sorted(list(candidate_texts))