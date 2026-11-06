"""
Cryptographic utilities for SAFELABS evidence management.

Provides AES-256-GCM authenticated encryption for evidence images at rest,
and SHA-256 integrity hashing for chain-of-custody verification.
"""
import os
import hashlib
import secrets
import base64

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ---------------------------------------------------------------------------
# Key Management
# ---------------------------------------------------------------------------

def generate_evidence_key() -> str:
    """
    Generate a new 256-bit AES key and return it as a URL-safe base64 string.
    Store the returned value in SAFELABS_EVIDENCE_KEY environment variable.
    """
    raw_key = secrets.token_bytes(32)  # 256 bits
    return base64.urlsafe_b64encode(raw_key).decode('utf-8')


def _load_evidence_key() -> bytes:
    """
    Load the AES-256 evidence encryption key from environment.
    Returns raw 32-byte key suitable for AESGCM.

    Raises:
        RuntimeError: If SAFELABS_EVIDENCE_KEY is not set.
    """
    encoded_key = os.environ.get('SAFELABS_EVIDENCE_KEY')
    if not encoded_key:
        raise RuntimeError(
            'SAFELABS_EVIDENCE_KEY environment variable is not set. '
            'Generate one with: python -c '
            '"from crypto_utils import generate_evidence_key; '
            'print(generate_evidence_key())"'
        )
    return base64.urlsafe_b64decode(encoded_key.encode('utf-8'))


# ---------------------------------------------------------------------------
# AES-256-GCM Encryption / Decryption
# ---------------------------------------------------------------------------

def encrypt_evidence(plaintext_bytes: bytes) -> bytes:
    """
    Encrypt evidence data using AES-256-GCM.

    The output format is:   nonce (12 bytes) || ciphertext+tag
    GCM authentication tag ensures tamper detection on decryption.

    Parameters:
        plaintext_bytes: Raw image bytes (e.g., JPEG data).

    Returns:
        Encrypted bytes (nonce prepended).
    """
    key = _load_evidence_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)
    return nonce + ciphertext


def decrypt_evidence(encrypted_bytes: bytes) -> bytes:
    """
    Decrypt AES-256-GCM encrypted evidence data.

    Parameters:
        encrypted_bytes: Encrypted data (nonce || ciphertext+tag).

    Returns:
        Decrypted plaintext bytes.

    Raises:
        cryptography.exceptions.InvalidTag: If data has been tampered with.
        RuntimeError: If encryption key is not configured.
    """
    key = _load_evidence_key()
    aesgcm = AESGCM(key)
    nonce = encrypted_bytes[:12]
    ciphertext = encrypted_bytes[12:]
    return aesgcm.decrypt(nonce, ciphertext, None)


# ---------------------------------------------------------------------------
# SHA-256 Integrity Hashing
# ---------------------------------------------------------------------------

def compute_evidence_hash(data: bytes) -> str:
    """
    Compute SHA-256 hash of evidence data for chain-of-custody integrity.

    The hash is computed on the *plaintext* image data before encryption,
    so it can be used to verify the decrypted output matches the original.

    Parameters:
        data: Raw image bytes.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(data).hexdigest()


def verify_evidence_integrity(data: bytes, expected_hash: str) -> bool:
    """
    Verify that decrypted evidence data matches its stored hash.

    Parameters:
        data: Decrypted image bytes.
        expected_hash: SHA-256 hex string stored at capture time.

    Returns:
        True if hashes match, False if evidence has been tampered with.
    """
    return compute_evidence_hash(data) == expected_hash
