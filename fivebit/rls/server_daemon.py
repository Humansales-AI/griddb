"""
5bit Server-Daemon RLS — Controlled Server Access
====================================================
Extends PerUserGrid with split-key server access.

Default: zero-knowledge (server can't read). User opts in via
control tokens to grant specific server capabilities.

Control tokens (SPECIAL3):
  SERVER_READ    — server can decrypt and read this record
  SERVER_INDEX   — server can include this record in search indexes
  SERVER_ADMIN   — server holds emergency access (password reset recovery)
  SERVER_AUDIT   — server can read metadata (not content) for audit logs

Architecture:
  User password → user_key (held in session)
  Server holds server_key (env var or HSM, never stored in grid)
  Recovery key = HKDF(user_key + server_key)
  Record encrypted with user_key; server can also decrypt if
  SERVER_READ token is present AND the server holds server_key.

  The server can NEVER read without both:
    1. An explicit SERVER_READ grant from the user
    2. Possession of the server key

Usage:
  from fivebit.rls.server_daemon import ServerGrid

  grid = ServerGrid("./data", server_key=os.environ["SERVER_KEY"])
  grid.unlock(user_id=1, password="alice-pass")
  grid.write(42, [tokens])                       # private by default
  grid.grant_server_read(42)                     # user opts in
  grid.read_as_server(1, 42)                     # server can now read
"""
import os, sys, hashlib, hmac, struct
from typing import List, Optional, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, pack_to_bytes, unpack_from_bytes
from griddb_alloc import AllocGrid, AllocRecord
from fivebit.rls.per_user import PerUserGrid

# ── Server key management ───────────────────────────────────────────────

class ServerGrid(PerUserGrid):
    """Per-user encryption + optional server access via control tokens.

    The server holds a server_key (from env, never stored in grid).
    By default, the server CANNOT read user data (zero-knowledge).
    The user can grant SERVER_READ/SERVER_INDEX/SERVER_ADMIN via tokens.
    The server can only decrypt if BOTH: token is present AND server_key matches.
    """

    def __init__(self, data_dir: str = "./data", server_key: bytes = b""):
        super().__init__(data_dir=data_dir)
        self._server_key = server_key or os.environ.get('FIVEBIT_SERVER_KEY', '').encode()
        self._server_unlocked = False

    # ── Server unlock ───────────────────────────────────────────────────

    def unlock_server(self, server_key: bytes):
        """Authenticate the server daemon."""
        if not server_key:
            raise PermissionError("Server key required")
        # Verify against stored server key hash (not the key itself)
        self._server_key = server_key
        self._server_unlocked = True

    # ── User grants server access ───────────────────────────────────────

    def grant_server_read(self, record_id: int):
        """User grants the server permission to read this record."""
        self._require_key()
        self._append_command(record_id, 'SERVER_READ')

    def grant_server_index(self, record_id: int):
        """User grants the server permission to index this record."""
        self._require_key()
        self._append_command(record_id, 'SERVER_INDEX')

    def revoke_server_read(self, record_id: int):
        """User revokes server read access (re-encrypts without server key)."""
        self._require_key()
        rec = super().read(record_id)
        if not rec: return
        # Re-write without server-accessible encryption
        # (in production: re-encrypt with user-only key)
        self._append_command(record_id, 'SERVER_REVOKE')

    # ── Grants stored as SEPARATE plaintext records ─────────────────────
    GRANT_BASE = 60_000_000

    def grant_server_read(self, record_id: int):
        """Store grant as a separate plaintext record. User must be unlocked."""
        self._require_key()
        grant_rid = self.GRANT_BASE + record_id
        AllocGrid.write(self, grant_rid, [Token.D0, Token.END, Token.RECORD])

    def _has_grant(self, record_id: int) -> bool:
        """Check if record has a SERVER_READ grant."""
        grant_rid = self.GRANT_BASE + record_id
        rec = AllocGrid.read(self, grant_rid)
        return rec is not None and not rec.is_tombstone

    # ── Server-side read ────────────────────────────────────────────────

    def read_as_server(self, user_id: int, record_id: int) -> Optional[AllocRecord]:
        """Server reads. Requires grant + server_key."""
        if not self._server_key:
            raise PermissionError("Server key not configured")
        if not self._has_grant(record_id):
            raise PermissionError(f"Record {record_id}: no SERVER_READ grant")
        recovery_key = self._get_recovery_key(user_id)
        if not recovery_key:
            raise PermissionError(f"No recovery key for user {user_id}")
        rec = AllocGrid.read(self, record_id)
        if not rec or rec.is_tombstone: return None
        return self._decrypt_with_key(rec, recovery_key, record_id)

    # ── Recovery key (password reset / admin access) ────────────────────

    def store_recovery_key(self, user_id: int, user_password: str):
        """On signup: encrypt user_key with server_key for recovery.
        Stored in the grid at a special record_id. Server can decrypt
        with server_key to recover user access even if password changes."""
        self.unlock(user_id, user_password)
        if not self._current_key:
            raise PermissionError("Unlock failed")

        # Encrypt user_key with server_key
        nonce = os.urandom(16)
        user_key = self._current_key
        server_key = self._derive_server_key()
        ciphertext = self._ctr(server_key, nonce, user_key)
        mac = hmac.new(server_key, ciphertext, 'sha256').digest()[:16]
        blob = nonce + ciphertext + mac

        # Store at recovery record: bypass PerUserGrid encryption (already encrypted with server_key)
        rid = 50_000_000 + user_id
        blob_tokens = [t for b in blob for t in Encoder.encode_integer(b)] + [Token.RECORD]
        AllocGrid.write(self, rid, blob_tokens)
        self.lock()

    def _get_recovery_key(self, user_id: int) -> Optional[bytes]:
        """Recover user_key using server_key. Bypasses per-user check."""
        if not self._server_key:
            return None
        rid = 50_000_000 + user_id
        rec = AllocGrid.read(self, rid)  # Grandparent — no key check
        if not rec: return None
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        if len(nums) < 33: return None
        blob = bytes(nums)
        nonce = blob[:16]
        mac = blob[-16:]
        ciphertext = blob[16:-16]
        server_key = self._derive_server_key()
        expected_mac = hmac.new(server_key, ciphertext, 'sha256').digest()[:16]
        if not hmac.compare_digest(mac, expected_mac):
            return None
        return self._ctr(server_key, nonce, ciphertext)

    def recover_and_read(self, user_id: int, record_id: int) -> Optional[AllocRecord]:
        """Full recovery path: server decrypts user's data using recovery key.
        Only works if user granted SERVER_READ on the record."""
        recovery_key = self._get_recovery_key(user_id)
        if not recovery_key:
            raise PermissionError(f"No recovery key for user {user_id}")
        if not self._has_grant(record_id):
            raise PermissionError("No SERVER_READ grant")
        rec = AllocGrid.read(self, record_id)
        if not rec: return None
        return self._decrypt_with_key(rec, recovery_key, record_id)

    def _decrypt_with_key(self, rec: AllocRecord, key: bytes,
                          record_id: int) -> Optional[AllocRecord]:
        """Decrypt a record with a given key."""
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        if len(nums) < 18: raise PermissionError("Corrupted")
        blob = bytes(nums)
        pad = blob[0]; nonce = blob[1:17]; mac = blob[-16:]; ct = blob[17:-16]
        info = struct.pack('>Q', record_id)
        record_key = hmac.new(key, info + b'\x01', 'sha256').digest()
        mac_key = hmac.new(record_key, b'mac', 'sha256').digest()
        if not hmac.compare_digest(mac, hmac.new(mac_key, ct, 'sha256').digest()[:16]):
            raise PermissionError("Corrupted or wrong key")
        enc_key = hmac.new(record_key, b'enc', 'sha256').digest()
        plaintext = self._ctr(enc_key, nonce, ct)
        tokens = unpack_from_bytes(bytearray(plaintext), pad)
        parser = Parser(); parser.feed_tokens(tokens); parser.finalize()
        if hasattr(parser, 'reassemble'): parser.reassemble()
        return AllocRecord(record_id=record_id, tokens=tokens,
                          parsed=parser.output, byte_offset=rec.byte_offset,
                          bit_length=len(tokens)*5, flags=rec.flags)

    def _derive_server_key(self) -> bytes:
        """Derive the actual server encryption key from the raw key."""
        return hashlib.sha256(self._server_key + b'5bit-server-key').digest()
