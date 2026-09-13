import pytest
from cryptography.fernet import Fernet, InvalidToken

from app.adapters.encryption import decrypt_bytes, encrypt_bytes


def test_encrypt_then_decrypt_round_trips_to_the_original_bytes():
    key = Fernet.generate_key()
    original = b"some plaintext model bytes"

    encrypted = encrypt_bytes(original, key)
    decrypted = decrypt_bytes(encrypted, key)

    assert decrypted == original
    assert encrypted != original


def test_decrypt_raises_on_tampered_ciphertext():
    key = Fernet.generate_key()
    encrypted = bytearray(encrypt_bytes(b"some plaintext", key))
    encrypted[-5] ^= 0xFF  # flip bits in a byte near the end of the token

    with pytest.raises(InvalidToken):
        decrypt_bytes(bytes(encrypted), key)


def test_decrypt_raises_on_wrong_key():
    key_a = Fernet.generate_key()
    key_b = Fernet.generate_key()
    encrypted = encrypt_bytes(b"some plaintext", key_a)

    with pytest.raises(InvalidToken):
        decrypt_bytes(encrypted, key_b)
