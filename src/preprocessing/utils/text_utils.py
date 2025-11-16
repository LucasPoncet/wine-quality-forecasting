import re


def normalize_string(text: str | None) -> str:
    """Lowercase, remove punctuation and extra spaces."""
    if not text:
        return ""
    text = text.lower().strip().replace("-", " ")
    text = re.sub(r"[\"'’`()\[\],.;:!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_wine_name(name: str | None) -> str:
    """Remove years from wine names."""
    if not name:
        return ""
    name = re.sub(r"\b(19|20)\d{2}\b", "", name.lower())
    return normalize_string(name)
