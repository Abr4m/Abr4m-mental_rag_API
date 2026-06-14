"""
Core: Security
Centralised security utilities (token signing, hashing, rate-limit helpers).
Extend this module as authentication requirements grow.
"""
import hashlib
import hmac
import secrets

from app.core.config import get_settings


def generate_token(length: int = 32) -> str:
    """Return a cryptographically secure random URL-safe token."""
    return secrets.token_urlsafe(length)


def constant_time_compare(val1: str, val2: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    return hmac.compare_digest(val1.encode(), val2.encode())


def hash_value(value: str) -> str:
    """SHA-256 hash of *value* — use for non-reversible fingerprinting."""
    return hashlib.sha256(value.encode()).hexdigest()
