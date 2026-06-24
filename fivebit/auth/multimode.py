"""
5bit Multi-Mode Auth — User Chooses Zero-Knowledge or Managed
===============================================================
One signup API. Two modes. Same GridDB underneath.

  mode='zero'    → PerUserGrid (per-user password keys, server blind)
  mode='managed' → CryptoRLS  (shared app secret, server can read)

Tradeoff:
  Zero:     server can NEVER read. Password reset = data loss.
  Managed:  server CAN read (analytics, search, support). Password reset works.

Usage:
  from fivebit.auth.multimode import MultiModeGrid

  grid = MultiModeGrid("./data", app_secret=b'...')
  grid.signup(1, 'alice', mode='zero')      # alice gets zero-knowledge
  grid.signup(2, 'bob', mode='managed')     # bob gets managed
  grid.login(1, 'alice')                    # unlocks alice's key
"""
import os, sys, hashlib, hmac, struct
from typing import Optional, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord, pack_to_bytes, unpack_from_bytes
from griddb_alloc import AllocGrid, AllocRecord
from fivebit.rls.per_user import PerUserGrid
from fivebit.rls.crypto import CryptoRLS

MODE_RECORD = 80_000_000  # record_id to store mode

class MultiModeGrid:
    """Unified zero/managed auth. Same AllocGrid, different key derivation."""

    def __init__(self, data_dir: str = "./data", app_secret: bytes = b""):
        self.base = AllocGrid(data_dir=data_dir)
        self.app_secret = app_secret or os.urandom(32)
        self._zero = PerUserGrid(data_dir=data_dir)  # shares base grid? No — separate dirs
        self._managed = CryptoRLS(data_dir=data_dir)
        self._mode_cache: Dict[int, str] = {}
        self._active: Optional[str] = None  # 'zero' or 'managed'
        self._current_user: Optional[int] = None
        self._current_key: Optional[bytes] = None

    # ── Signup / Login ──────────────────────────────────────────────────

    def signup(self, user_id: int, password: str, mode: str = 'zero') -> bool:
        """Create user. mode='zero' or 'managed'."""
        if mode not in ('zero', 'managed'):
            raise ValueError("mode must be 'zero' or 'managed'")

        # Store mode in a plaintext record
        self.base.write(MODE_RECORD + user_id, [
            *Encoder.encode_integer(user_id),
            *Encoder.encode_word(mode.upper()),
            Token.RECORD,
        ])
        self._mode_cache[user_id] = mode

        # Derive and store key for the chosen mode
        salt = f"5bit-user-{user_id}".encode()
        user_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000, 32)

        if mode == 'zero':
            # Recovery code = 12-word backup (deterministic from key)
            recovery = self._generate_recovery(user_key)
            self.base.write(MODE_RECORD + user_id + 1, [
                *Encoder.encode_word(recovery),
                Token.RECORD,
            ])
        else:
            # Managed: encrypt user_key with app_secret for server access
            nonce = os.urandom(16)
            ciphertext = self._ctr(self.app_secret, nonce, user_key)
            mac = hmac.new(self.app_secret, ciphertext, 'sha256').digest()[:16]
            blob = nonce + ciphertext + mac
            blob_tokens = [t for b in blob for t in Encoder.encode_integer(b)] + [Token.RECORD]
            self.base.write(MODE_RECORD + user_id + 2, blob_tokens)

        return True

    def login(self, user_id: int, password: str) -> bool:
        """Unlock user's key. Works for both modes."""
        mode = self.get_mode(user_id)
        salt = f"5bit-user-{user_id}".encode()
        user_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000, 32)

        self._current_user = user_id
        self._current_key = user_key
        self._active = mode
        return True

    def lock(self):
        self._current_user = None
        self._current_key = None
        self._active = None

    # ── Mode queries ────────────────────────────────────────────────────

    def get_mode(self, user_id: int) -> str:
        """Read the user's mode from the grid."""
        if user_id in self._mode_cache:
            return self._mode_cache[user_id]
        rec = self.base.read(MODE_RECORD + user_id)
        if rec:
            words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
            mode = ''.join(words).lower()
            self._mode_cache[user_id] = mode
            return mode
        return 'zero'  # default

    def get_recovery_code(self, user_id: int) -> Optional[str]:
        """Get recovery code (zero mode only)."""
        rec = self.base.read(MODE_RECORD + user_id + 1)
        if rec:
            return ''.join(p.text for p in rec.parsed if isinstance(p, ParsedWord))
        return None

    # ── Storage (routes to correct backend) ────────────────────────────

    def write(self, record_id: int, tokens: List[Token]) -> int:
        self._require_login()
        if self._active == 'zero':
            return self._zero_write(record_id, tokens)
        else:
            return self._managed_write(record_id, tokens)

    def read(self, record_id: int) -> Optional[AllocRecord]:
        self._require_login()
        if self._active == 'zero':
            return self._zero_read(record_id)
        else:
            return self._managed_read(record_id)

    # ── Zero-knowledge backend ──────────────────────────────────────────

    def _zero_write(self, rid: int, tokens: List[Token]) -> int:
        info = struct.pack('>Q', rid)
        record_key = hmac.new(self._current_key, info + b'\x01', 'sha256').digest()
        enc_key = hmac.new(record_key, b'enc', 'sha256').digest()
        mac_key = hmac.new(record_key, b'mac', 'sha256').digest()
        packed, pad = pack_to_bytes(tokens)
        nonce = os.urandom(16)
        ct = self._ctr(enc_key, nonce, bytes(packed))
        mac = hmac.new(mac_key, ct, 'sha256').digest()[:16]
        blob = struct.pack('>B', pad) + nonce + ct + mac
        blob_tokens = [t for b in blob for t in Encoder.encode_integer(b)] + [Token.RECORD]
        return self.base.write(rid, blob_tokens)

    def _zero_read(self, rid: int) -> Optional[AllocRecord]:
        rec = self.base.read(rid)
        if not rec or rec.flags == 2: return None
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        if len(nums) < 18: raise PermissionError("Corrupted")
        blob = bytes(nums); pad = blob[0]; nonce = blob[1:17]; mac = blob[-16:]; ct = blob[17:-16]
        info = struct.pack('>Q', rid)
        record_key = hmac.new(self._current_key, info + b'\x01', 'sha256').digest()
        mac_key = hmac.new(record_key, b'mac', 'sha256').digest()
        if not hmac.compare_digest(mac, hmac.new(mac_key, ct, 'sha256').digest()[:16]):
            raise PermissionError("Wrong key")
        enc_key = hmac.new(record_key, b'enc', 'sha256').digest()
        plaintext = self._ctr(enc_key, nonce, ct)
        tokens = unpack_from_bytes(bytearray(plaintext), pad)
        p = Parser(); p.feed_tokens(tokens); p.finalize()
        if hasattr(p, 'reassemble'): p.reassemble()
        return AllocRecord(record_id=rid, tokens=tokens, parsed=p.output,
                          byte_offset=rec.byte_offset, bit_length=len(tokens)*5, flags=rec.flags)

    # ── Managed backend ─────────────────────────────────────────────────

    def _managed_write(self, rid: int, tokens: List[Token]) -> int:
        from fivebit.rls.crypto import derive_key, _ctr_encrypt
        packed, pad = pack_to_bytes(tokens)
        key = derive_key(self._current_user, self.app_secret)
        encrypted = _ctr_encrypt(key, bytes(packed))
        mac = hmac.new(key, encrypted, 'sha256').digest()[:16]
        blob = struct.pack('>B', pad) + encrypted + mac
        blob_tokens = [t for b in blob for t in Encoder.encode_integer(b)] + [Token.RECORD]
        return self.base.write(rid, blob_tokens)

    def _managed_read(self, rid: int) -> Optional[AllocRecord]:
        from fivebit.rls.crypto import derive_key, _ctr_decrypt
        rec = self.base.read(rid)
        if not rec or rec.flags == 2: return None
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        if len(nums) < 18: raise PermissionError("Corrupted")
        blob = bytes(nums); pad = blob[0]; encrypted = blob[1:-16]; mac = blob[-16:]
        key = derive_key(self._current_user, self.app_secret)
        expected = hmac.new(key, encrypted, 'sha256').digest()[:16]
        if not hmac.compare_digest(mac, expected): raise PermissionError("Wrong key")
        plaintext = _ctr_decrypt(key, encrypted)
        tokens = unpack_from_bytes(bytearray(plaintext), pad)
        p = Parser(); p.feed_tokens(tokens); p.finalize()
        if hasattr(p, 'reassemble'): p.reassemble()
        return AllocRecord(record_id=rid, tokens=tokens, parsed=p.output,
                          byte_offset=rec.byte_offset, bit_length=len(tokens)*5, flags=rec.flags)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _require_login(self):
        if not self._current_key:
            raise PermissionError("Call login() first")

    def _generate_recovery(self, key: bytes) -> str:
        """Generate 12 recovery words from the key (deterministic)."""
        words = [
            'abandon','ability','able','about','above','absent','absorb','abstract',
            'acoustic','acquire','across','act','action','actor','actress','actual',
            'adapt','add','addict','address','adjust','admit','adult','advance',
        ]
        h = hashlib.sha256(key + b'recovery').digest()
        indices = [int.from_bytes(h[i*2:i*2+2], 'big') % len(words) for i in range(12)]
        return ' '.join(words[i] for i in indices)

    def _ctr(self, key: bytes, nonce: bytes, data: bytes) -> bytes:
        result = bytearray(len(data))
        for i in range(0, len(data), 32):
            ctr = nonce + struct.pack('>Q', i // 32)
            ks = hmac.new(key, ctr, 'sha256').digest()
            for j in range(i, min(i+32, len(data))): result[j] = data[j] ^ ks[j-i]
        return bytes(result)

    def close(self): self.base.close()
