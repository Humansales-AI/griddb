"""
5bit Command RLS — Token-Level Permission Representation
==========================================================
SPECIAL3 commands (AUTH, GRANT, REVOKE) embedded in the token stream.
These are a *representation* of intended permissions, not enforcement.

Honest: a raw AllocGrid on the same directory reads everything.
The AUTH token says "user 1 owns this" — it does not prevent reading.
For actual enforcement, combine with CryptoRLS (v5.3) for encryption.

Usage:
  from fivebit.rls.commands import CommandRLS
  grid = CommandRLS("./data")
  grid.write_owned(42, user_id=1, tokens=[...])    # AUTH tag prepended
  grid.read(42, user_id=1)  # ✓ checks AUTH tag
  grid.read(42, user_id=2)  # ✗ PermissionDenied (in-process only)
"""
import os, sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord, CMD_AUTH, CMD_NAMES
from griddb_alloc import AllocGrid, AllocRecord


class PermissionDenied(Exception):
    pass


class CommandRLS(AllocGrid):
    """RLS via SPECIAL3 control commands. Extends AllocGrid."""

    def __init__(self, data_dir: str = "./data"):
        super().__init__(data_dir=data_dir)
        self._bypass_user: Optional[int] = None

    def write_owned(self, record_id: int, user_id: int, tokens: List[Token]) -> int:
        """Write a record owned by user_id. Prepends AUTH command."""
        cmd_tokens = Encoder.encode_command('AUTH', user_id)
        return super().write(record_id, cmd_tokens + tokens)

    def read(self, record_id: int, user_id: int = 0) -> Optional[AllocRecord]:
        """Read with RLS check. Parses AUTH + GRANT commands."""
        rec = super().read(record_id)
        if not rec or rec.is_tombstone: return None
        perms = self._parse_commands(rec.tokens)
        owner = perms.get('owner')
        grantees = perms.get('grantees', set())
        if owner is not None and user_id != owner and user_id not in grantees and self._bypass_user is None:
            raise PermissionDenied(
                f"CommandRLS: user {user_id} cannot read record {record_id} (owner={owner})")
        return rec

    def _parse_commands(self, tokens: List[Token]) -> dict:
        """Parse SPECIAL3 commands to extract owner + grantees."""
        result = {'owner': None, 'grantees': set()}
        parser = Parser()
        for t in tokens: parser.feed(t)
        parser.finalize()
        cmd = None
        for p in parser.output:
            if isinstance(p, dict) and p.get('type') == 'command':
                cmd = p.get('cmd')
                continue
            if cmd and isinstance(p, ParsedNumber):
                if cmd == 'AUTH': result['owner'] = p.value
                elif cmd == 'GRANT_R': result['grantees'].add(p.value)
                elif cmd == 'REVOKE': result['grantees'].discard(p.value)
                cmd = None
        return result

    def grant_read(self, record_id: int, user_id: int, grantee_id: int):
        """Append a GRANT_R command to the record."""
        rec = super().read(record_id)
        if not rec: return
        cmd = Encoder.encode_command('GRANT_R', grantee_id)
        self.write_owned(record_id, user_id, rec.tokens + cmd)

    def revoke_read(self, record_id: int, user_id: int, target_id: int):
        """Append a REVOKE command to the record."""
        rec = super().read(record_id)
        if not rec: return
        cmd = Encoder.encode_command('REVOKE', target_id)
        self.write_owned(record_id, user_id, rec.tokens + cmd)

    def delete(self, record_id: int, user_id: int = 0) -> bool:
        self.read(record_id, user_id)  # Verify ownership
        return super().delete(record_id)

    def as_admin(self):
        return _AC(self)


class _AC:
    def __init__(self, e): self.e = e
    def __enter__(self): self.e._bypass_user = -1; return self.e
    def __exit__(self, *a): self.e._bypass_user = None
