"""
5bit Auth — User & Session Management
======================================
Out-of-the-box authentication on top of the 5bit core.
Users stored as separate field records (email, hash, name) at known offsets.

Usage:
  from fivebit.auth import AuthStore
  auth = AuthStore("./data")
  uid = auth.signup("alice@demo.com", "password123", "Alice")
  session = auth.login("alice@demo.com", "password123")
"""
import os, sys, hashlib, secrets, time
from typing import Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid

SESSION_BASE = 10_000_000
FIELD_STRIDE = 100  # user_id * STRIDE + field_offset

class AuthStore:
    def __init__(self, data_dir: str = "./auth_data"):
        self.grid = AllocGrid(data_dir=data_dir)

    def _hash(self, pw: str) -> str:
        """Returns salt$hex_hash. Use letters-only mapping for digits."""
        salt = secrets.token_hex(16)
        h = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 100_000).hex()
        return salt + '$' + h

    def _encode_hash(self, h: str) -> list:
        """Encode a hash string as NUM tokens (ASCII bytes). Survives any context."""
        tokens = []
        for ch in h:
            tokens.extend(Encoder.encode_integer(ord(ch)))
        tokens.append(Token.RECORD)
        return tokens

    def _decode_hash(self, tokens: list) -> str:
        """Decode NUM tokens back to a hash string."""
        vals = []
        for t in tokens:
            if t == Token.RECORD: continue
            if hasattr(t, 'value'): vals.append(t.value)
            elif isinstance(t, int) and 0 <= t <= 9: vals.append(t)
        # Parse numbers: each NUM is an ASCII code
        p = Parser()
        for t in tokens:
            if t != Token.RECORD: p.feed(t)
        p.finalize()
        nums = [x.value for x in p.output if isinstance(x, ParsedNumber)]
        return ''.join(chr(n) for n in nums)

    def _verify(self, pw: str, stored: str) -> bool:
        try:
            s, h = stored.split('$')
            return h == hashlib.pbkdf2_hmac('sha256', pw.encode(), s.encode(), 100_000).hex()
        except: return False

    def _rid(self, uid: int, field: int) -> int: return uid * FIELD_STRIDE + field

    def _read_field(self, uid: int, field: int) -> str:
        rec = self.grid.read(self._rid(uid, field))
        if not rec: return ''
        if field == 2:  # Hash stored as NUM tokens
            nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
            # First two nums are uid + fieldId, rest are ASCII codes
            return ''.join(chr(n) for n in nums[2:]) if len(nums) > 2 else ''
        return ''.join(p.text for p in rec.parsed if isinstance(p, ParsedWord))

    def _write_field(self, uid: int, field: int, value: str):
        if field == 2:  # Hash: store as NUM tokens to survive context switching
            tokens = [*Encoder.encode_integer(uid), *Encoder.encode_integer(field),
                      *self._encode_hash(value)]
        else:
            tokens = [*Encoder.encode_integer(uid), *Encoder.encode_integer(field),
                      *Encoder.encode_word(value), Token.RECORD]
        self.grid.write(self._rid(uid, field), tokens)

    def signup(self, email: str, password: str, name: str = "") -> int:
        uid = self._next_uid()
        self._write_field(uid, 1, email)
        self._write_field(uid, 2, self._hash(password))
        self._write_field(uid, 3, name)
        return uid

    def login(self, email: str, password: str) -> Optional[object]:
        user = self._by_email(email)
        if not user or not self._verify(password, user['hash']): return None
        return self._create_session(user['id'])

    def get_user(self, uid: int) -> Optional[Dict]:
        email = self._read_field(uid, 1)
        if not email: return None
        return {'id': uid, 'email': email, 'hash': self._read_field(uid, 2),
                'name': self._read_field(uid, 3)}

    def _by_email(self, email: str) -> Optional[Dict]:
        for uid in range(1, self._next_uid()):
            u = self.get_user(uid)
            if u and u['email'] == email: return u
        return None

    def _next_uid(self) -> int:
        uid = 1
        while self._read_field(uid, 1): uid += 1
        return uid

    class Session:
        def __init__(self, token: str, user_id: int, expires: float):
            self.token = token; self.user_id = user_id; self.expires_at = expires
        @property
        def is_expired(self) -> bool: return time.time() > self.expires_at

    def _create_session(self, uid: int):
        token = secrets.token_hex(32)
        rid = SESSION_BASE + (hash(token) & 0xFFFFF)
        # Store token as NUM tokens (ASCII bytes) to survive context switching
        t = [Encoder.encode_integer(ord(c)) for c in token]
        tokens = [tkn for sub in t for tkn in sub]  # flatten
        tokens.extend(Encoder.encode_integer(uid))
        tokens.extend(Encoder.encode_integer(int(time.time()+86400)))
        tokens.append(Token.RECORD)
        self.grid.write(rid, tokens)
        return AuthStore.Session(token, uid, time.time() + 86400)

    def verify_session(self, token: str) -> Optional[int]:
        rid = SESSION_BASE + (hash(token) & 0xFFFFF)
        rec = self.grid.read(rid)
        if not rec: return None
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        if len(nums) < 3: return None
        # Reconstruct token from ASCII codes (first N nums)
        token_len = len(token)
        stored = ''.join(chr(n) for n in nums[:token_len])
        if stored == token and time.time() < nums[-1]:
            return nums[-2]  # userId is second-to-last
        return None

    def logout(self, token: str):
        self.grid.delete(SESSION_BASE + (hash(token) & 0xFFFFF))

    def close(self): self.grid.close()
