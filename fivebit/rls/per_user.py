"""
5bit Per-User RLS — True Multi-Tenant Isolation
==================================================
No shared app secret. Each user's records are encrypted with a key
derived from THEIR password — not from a master secret.

The engine literally cannot decrypt user B's records without user B's
credential. Even root on the server sees only ciphertext. This is the
same architecture as password managers and E2E encrypted apps.

Architecture:
  user_password → PBKDF2 → user_key (held in session, never on disk)
  record_key = HKDF(user_key, record_id)
  record encrypted with record_key

Cross-user access is cryptographically impossible — the engine
doesn't have the keys.

Usage:
  from fivebit.rls.per_user import PerUserGrid

  grid = PerUserGrid("./data")
  grid.unlock(user_id=1, password="alice-secret")     # derive key
  grid.write(42, [tokens])                             # auto-encrypts
  grid.read(42)                                        # auto-decrypts
  grid.lock()                                          # wipe keys from memory
"""
import os, sys, hashlib, hmac, struct
from typing import List, Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, pack_to_bytes, unpack_from_bytes
from griddb_alloc import AllocGrid, AllocRecord


class PerUserGrid(AllocGrid):
    """True multi-tenant isolation — per-user keys, no shared secret.

    Each user's records are encrypted with a key derived from their
    password. The engine never stores the password or the derived key
    on disk. Cross-user decryption is cryptographically impossible.
    """

    def __init__(self, data_dir: str = "./data"):
        super().__init__(data_dir=data_dir)
        self._current_user: Optional[int] = None
        self._current_key: Optional[bytes] = None
        self._key_cache: Dict[int, bytes] = {}  # session cache only

    # ── Key management ──────────────────────────────────────────────────

    def unlock(self, user_id: int, password: str):
        """Derive the user's key from their password. Held in memory only."""
        salt = f"5bit-user-{user_id}".encode()
        self._current_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000, 32)
        self._current_user = user_id
        self._key_cache[user_id] = self._current_key

    def lock(self):
        """Wipe keys from memory."""
        self._current_user = None
        self._current_key = None
        self._key_cache.clear()

    @property
    def is_unlocked(self) -> bool:
        return self._current_key is not None

    def _require_key(self):
        if not self._current_key:
            raise PermissionError("Call unlock(user_id, password) first")

    # ── Storage (auto-encrypt/decrypt) ──────────────────────────────────

    def write(self, record_id: int, tokens: List[Token]) -> int:
        """Write tokens encrypted with the current user's key."""
        self._require_key()
        # HKDF-expand: per-record key from user_key + record_id
        # Uses HMAC-SHA256 as the PRF (stdlib HKDF)
        info = struct.pack('>Q', record_id)
        record_key = hmac.new(self._current_key, info + b'\x01', 'sha256').digest()

        # Encrypt with HMAC-CTR. Separate enc + MAC keys via HKDF.
        enc_key = hmac.new(record_key, b'enc', 'sha256').digest()
        mac_key = hmac.new(record_key, b'mac', 'sha256').digest()
        packed, pad = pack_to_bytes(tokens)
        nonce = os.urandom(16)
        ciphertext = self._ctr(enc_key, nonce, bytes(packed))
        mac = hmac.new(mac_key, ciphertext, 'sha256').digest()[:16]
        blob = struct.pack('>B', pad) + nonce + ciphertext + mac

        # Store blob as NUM tokens
        blob_tokens: List[Token] = []
        for b in blob:
            blob_tokens.extend(Encoder.encode_integer(b))
        blob_tokens.append(Token.RECORD)
        return super().write(record_id, blob_tokens)

    def read(self, record_id: int) -> Optional[AllocRecord]:
        """Read and decrypt with the current user's key. Wrong key → fail."""
        self._require_key()
        rec = super().read(record_id)
        if not rec or rec.is_tombstone:
            return None

        # Extract blob
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        if len(nums) < 18:
            raise PermissionError("Record corrupted or not encrypted")

        blob = bytes(nums)
        pad = blob[0]
        nonce = blob[1:17]
        mac = blob[-16:]
        ciphertext = blob[17:-16]

        # Re-derive record key via HKDF, then enc + mac subkeys
        info = struct.pack('>Q', record_id)
        record_key = hmac.new(self._current_key, info + b'\x01', 'sha256').digest()
        mac_key = hmac.new(record_key, b'mac', 'sha256').digest()
        expected_mac = hmac.new(mac_key, ciphertext, 'sha256').digest()[:16]
        if not hmac.compare_digest(mac, expected_mac):
            raise PermissionError(
                f"Record {record_id}: wrong key — belongs to a different user")

        # Decrypt with separate encryption key
        enc_key = hmac.new(record_key, b'enc', 'sha256').digest()
        plaintext = self._ctr(enc_key, nonce, ciphertext)
        tokens = unpack_from_bytes(bytearray(plaintext), pad)
        parser = Parser()
        parser.feed_tokens(tokens); parser.finalize()
        if hasattr(parser, 'reassemble'): parser.reassemble()

        return AllocRecord(record_id=record_id, tokens=tokens,
                          parsed=parser.output, byte_offset=rec.byte_offset,
                          bit_length=len(tokens)*5, flags=rec.flags)

    def delete(self, record_id: int) -> bool:
        self._require_key()
        self.read(record_id)  # Verify ownership via decryption
        return super().delete(record_id)

    def _ctr(self, key: bytes, nonce: bytes, data: bytes) -> bytes:
        """CTR mode with HMAC-SHA256 as PRF."""
        result = bytearray(len(data))
        for i in range(0, len(data), 32):
            ctr = nonce + struct.pack('>Q', i // 32)
            ks = hmac.new(key, ctr, 'sha256').digest()
            end = min(i + 32, len(data))
            for j in range(i, end):
                result[j] = data[j] ^ ks[j - i]
        return bytes(result)
