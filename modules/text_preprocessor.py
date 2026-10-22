"""Text normalization and chunking utilities."""
import re


def preprocess_for_embedding(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text: str, max_chars: int = 900) -> list[str]:
    """Split text on paragraph/sentence boundaries, with safe hard splitting."""
    text = preprocess_for_embedding(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
