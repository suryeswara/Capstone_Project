"""
MedVerify AI - Security & Authentication Helper Module

Implements:
1. PBKDF2-HMAC-SHA256 password hashing & verification
2. Cryptographically signed JWT access token creation & verification
"""

import os
import hmac
import json
import time
import base64
import hashlib
from typing import Optional, Dict

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "medverify-super-secret-jwt-key-2026-phase1")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 86400 * 7  # 7 days


# ---------------------------------------------------------------------------
# PASSWORD HASHING (PBKDF2-HMAC-SHA256 with random 16-byte salt)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    salt_hex = salt.hex()
    key_hex = key.hex()
    return f"pbkdf2_sha256$100000${salt_hex}${key_hex}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against stored hash string."""
    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_key = bytes.fromhex(parts[3])
        actual_key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual_key, expected_key)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT TOKEN GENERATION & VERIFICATION (HMAC-SHA256)
# ---------------------------------------------------------------------------

def _b64_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64_decode(data: str) -> bytes:
    """Base64url decode with padding restoration."""
    padded = data + "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(padded)


def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
    """Create a signed JWT access token."""
    header = {"alg": "HS256", "typ": "JWT"}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_b64 = _b64_encode(header_bytes)

    expire = time.time() + (expires_delta or ACCESS_TOKEN_EXPIRE_SECONDS)
    payload = {**data, "exp": int(expire)}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64_encode(payload_bytes)

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = _b64_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str) -> Optional[dict]:
    """Verify signature and expiration of JWT token, returning payload dict if valid."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        actual_sig = _b64_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        # Parse payload
        payload_bytes = _b64_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        # Verify expiration
        exp = payload.get("exp")
        if exp and time.time() > exp:
            return None

        return payload
    except Exception:
        return None
