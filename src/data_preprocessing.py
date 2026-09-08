"""
Text preprocessing utilities shared by training and inference.
"""
import re
import string


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/punctuation/extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                       # HTML tags
    text = re.sub(r"\S+@\S+", " ", text)                     # emails
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                         # numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


SENSATIONAL_PATTERNS = [
    r"\byou won'?t believe\b", r"\bshocking\b", r"\bthey don'?t want you to know\b",
    r"\bmiracle\b", r"\bsecretly\b", r"\bbanned\b", r"\bcover[- ]?up\b",
    r"\bwhat happens next\b", r"\bclick to find out\b", r"\bone weird trick\b",
    r"\binsider claims\b", r"\bexperts baffled\b", r"\bgovernment doesn'?t want\b",
    r"\bcure[d]? overnight\b", r"\bviral post\b", r"\bwithout any source\b",
]


def sensationalism_score(raw_text: str) -> float:
    """Cheap heuristic: fraction of known clickbait/sensational patterns present,
    plus punctuation-based signals (ALL CAPS words, excessive '!'). Range ~0-1+.
    Used as one signal inside the LLM-analysis fallback, not the primary model."""
    if not raw_text:
        return 0.0
    text_lower = raw_text.lower()
    hits = sum(1 for pat in SENSATIONAL_PATTERNS if re.search(pat, text_lower))
    exclaim = raw_text.count("!")
    caps_words = len(re.findall(r"\b[A-Z]{4,}\b", raw_text))
    score = hits * 0.15 + min(exclaim, 5) * 0.05 + min(caps_words, 5) * 0.05
    return round(min(score, 1.5), 3)
