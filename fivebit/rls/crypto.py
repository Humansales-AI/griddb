"""
5bit RLS Crypto — Encrypted Storage-Layer RLS (stdlib only)
=============================================================
Encrypts every record with a per-user key derived via PBKDF2-SHA256.
Uses AES-style CTR mode via hashlib + HMAC authentication.
Zero external dependencies. Python stdlib only.

Another process opening the data directory sees ciphertext.
Owner ID is never stored in plaintext.

Usage:
  from fivebit.rls.crypto import CryptoRLS
  grid = CryptoRLS("./data")
  grid.write(42, user_id=1, secret_key=b'...', tokens=[...])
  grid.read(42, user_id=1, secret_key=b'...')  # ✓
  grid.read(42, user_id=2, secret_key=b'...')  # ✗ PermissionDenied
"""
import os, sys, struct, hashlib, hmac
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, pack_to_bytes, unpack_from_bytes
from griddb_alloc import AllocGrid, AllocRecord
from fivebit.rls.engine import PermissionDenied

# Key cache — PBKDF2 only runs once per user, not per read/write
_key_cache: dict = {}

def derive_key(user_id: int, secret: bytes) -> bytes:
    """Derive a 32-byte key from user_id + secret. Cached after first derivation."""
    cache_key = (user_id, secret)
    if cache_key not in _key_cache:
        _key_cache[cache_key] = hashlib.pbkdf2_hmac('sha256', secret, str(user_id).encode(), 100_000, 32)
    return _key_cache[cache_key]

def _ctr_encrypt(key: bytes, plaintext: bytes) -> bytes:
    """CTR-mode encryption using HMAC-SHA256 as the PRF (standard construction)."""
    nonce = os.urandom(16)
    result = bytearray(nonce)
    counter = 0
    for i in range(0, len(plaintext), 32):
        ctr_block = nonce + struct.pack('>Q', counter)
        keystream = hmac.new(key, ctr_block, 'sha256').digest()
        chunk = plaintext[i:i+32]
        result.extend(b ^ k for b, k in zip(chunk, keystream[:len(chunk)]))
        counter += 1
    return bytes(result)

def _ctr_decrypt(key: bytes, ciphertext: bytes) -> bytes:
    """Decrypt CTR-mode ciphertext. CTR is symmetric — same as encrypt."""
    nonce = ciphertext[:16]
    data = ciphertext[16:]
    result = bytearray()
    counter = 0
    for i in range(0, len(data), 32):
        ctr_block = nonce + struct.pack('>Q', counter)
        keystream = hmac.new(key, ctr_block, 'sha256').digest()
        chunk = data[i:i+32]
        result.extend(b ^ k for b, k in zip(chunk, keystream[:len(chunk)]))
        counter += 1
    return bytes(result)


class CryptoRLS(AllocGrid):
    """Encrypted RLS extends AllocGrid. Files on disk = ciphertext only."""

    def __init__(self, data_dir: str = "./data"):
        super().__init__(data_dir=data_dir)
        self._bypass_user: Optional[int] = None

    def write(self, record_id: int, user_id: int, secret_key: bytes,
              tokens: List[Token]) -> int:
        packed, pad = pack_to_bytes(tokens)
        key = derive_key(user_id, secret_key)
        encrypted = _ctr_encrypt(key, bytes(packed))
        # HMAC for authentication
        mac = hmac.new(key, encrypted, 'sha256').digest()[:16]
        blob = struct.pack('>B', pad) + encrypted + mac
        blob_tokens = [t for b in blob for t in Encoder.encode_integer(b)] + [Token.RECORD]
        return super().write(record_id, blob_tokens)

    def read(self, record_id: int, user_id: int,
             secret_key: bytes) -> Optional[AllocRecord]:
        rec = super().read(record_id)
        if not rec or rec.is_tombstone: return None
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        if len(nums) < 18: raise PermissionDenied("Corrupted record")
        blob = bytes(nums)
        pad = blob[0]; encrypted = blob[1:-16]; mac = blob[-16:]
        key = derive_key(user_id, secret_key)
        # Verify HMAC
        expected_mac = hmac.new(key, encrypted, 'sha256').digest()[:16]
        if not hmac.compare_digest(mac, expected_mac):
            raise PermissionDenied("Wrong key or corrupted data")
        plaintext = _ctr_decrypt(key, encrypted)
        tokens = unpack_from_bytes(bytearray(plaintext), pad)
        parser = Parser()
        parser.feed_tokens(tokens); parser.finalize()
        if hasattr(parser, 'reassemble'): parser.reassemble()
        return AllocRecord(record_id=record_id, tokens=tokens,
                          parsed=parser.output, byte_offset=rec.byte_offset,
                          bit_length=len(tokens)*5, flags=rec.flags)

    def delete(self, record_id: int, user_id: int, secret_key: bytes) -> bool:
        self.read(record_id, user_id, secret_key)  # Verify ownership
        return super().delete(record_id)

    def as_admin(self): return _AdminContext(self)

class _AdminContext:
    def __init__(self, e): self.e = e
    def __enter__(self): self.e._bypass_user = -1; return self.e
    def __exit__(self, *a): self.e._bypass_user = None
