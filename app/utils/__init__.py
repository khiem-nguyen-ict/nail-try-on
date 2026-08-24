import hashlib


def image_id(image_bytes: bytes) -> str:
    """Return a short SHA256 hex id for image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()[:16]
