"""
Application-level encryption for account numbers at rest.

Why encrypt in the application layer rather than relying only on Postgres-level
encryption (e.g. pgcrypto, or disk-level encryption): disk/volume encryption protects
against physical media theft but not against a stolen database credential or a SQL
injection in some other part of a larger system — the account_number column itself
is unreadable even to someone who can run arbitrary SELECTs with valid DB credentials
but doesn't have SECRET_KEY. This is defense in depth, not a replacement for
transport encryption (TLS to Postgres) or infrastructure-level encryption, both of
which a real deployment should also have.

Fernet (symmetric, authenticated encryption) is used rather than a bespoke cipher —
it's the standard `cryptography` library's recommended choice for "encrypt this
value, decrypt it later, with one shared key," and it includes built-in integrity
verification (a tampered ciphertext fails to decrypt rather than silently decrypting
to garbage).

Key derivation: SECRET_KEY (a human-set string in .env) is not itself a valid Fernet
key (Fernet requires 32 url-safe base64-encoded bytes) — we derive one deterministically
via SHA-256, so operators only need to manage one secret, not two.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.exceptions import AppError


class DecryptionError(AppError):
    error_code = "decryption_error"
    http_status = 500


def _derive_fernet_key(secret_key: str) -> bytes:
    digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


class FieldEncryptor:
    """Stateless-per-key wrapper around Fernet. One instance is built from
    settings.secret_key at repository-construction time and reused for every
    encrypt/decrypt call in that process — Fernet itself is cheap to use repeatedly."""

    def __init__(self, secret_key: str) -> None:
        self._fernet = Fernet(_derive_fernet_key(secret_key))

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise DecryptionError(
                "Failed to decrypt field — ciphertext is invalid or SECRET_KEY has changed."
            ) from exc
