"""Input validation and sanitization for the PRD Generator API.

Provides lightweight input validation to protect against:
- Prompt injection attempts
- Oversized inputs (truncation)
- Empty/malformed inputs
- Special character handling
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_PRODUCT_IDEA_LENGTH = 500
MAX_SUPPLEMENTARY_LENGTH = 3000
MAX_FEEDBACK_LENGTH = 2000

# Patterns that indicate potential prompt injection
_INJECTION_PATTERNS = [
    "ignore all previous",
    "ignore previous instructions",
    "forget all",
    "you are now",
    "system prompt",
    "<|im_start|>",
    "<|im_end|>",
    "### system",
    "### assistant",
    "<system>",
    "</system>",
]


def validate_product_idea(text: str, max_length: int = MAX_PRODUCT_IDEA_LENGTH) -> str:
    """Validate and sanitize the product idea input.

    Returns the cleaned text, or an empty string if input is invalid.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    if len(text) > max_length:
        logger.warning("Product idea truncated from %d to %d chars", len(text), max_length)
        text = text[:max_length]

    return text


def validate_supplementary(text: str, max_length: int = MAX_SUPPLEMENTARY_LENGTH) -> str:
    """Validate and truncate supplementary info."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > max_length:
        logger.warning("Supplementary info truncated from %d to %d chars", len(text), max_length)
        text = text[:max_length]
    return text


def validate_feedback(text: str, max_length: int = MAX_FEEDBACK_LENGTH) -> str:
    """Validate revision feedback input."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text


def check_prompt_injection(text: str) -> list[str]:
    """Check for potential prompt injection patterns.

    Returns a list of suspicious patterns found (empty list = clean).
    """
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for pattern in _INJECTION_PATTERNS:
        if pattern in text_lower:
            found.append(pattern)
    if found:
        logger.warning("Potential prompt injection detected, patterns: %s", found)
    return found


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename, removing path traversal and dangerous characters."""
    import re
    if not filename:
        return "untitled"
    # Remove path separators and traversal
    filename = filename.replace("\\", "_").replace("/", "_")
    filename = filename.replace("..", "__")
    # Keep only safe characters
    filename = re.sub(r'[^\w\.\-]', '_', filename)
    # Remove leading dots (hidden files)
    filename = filename.lstrip(".")
    if not filename:
        return "untitled"
    return filename[:200]  # Max filename length
