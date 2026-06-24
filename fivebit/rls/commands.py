"""
5bit Command RLS — Token-Level Access Control
===============================================
RLS enforced by SPECIAL3 control commands in the token fabric.
No encryption overhead. No wrapper. Just token comparisons at parse time.

A record with AUTH(user_id) in its token stream can only be read
by that user. The parser checks on every read.

Usage:
  from fivebit.rls.commands import CommandRLS
  grid = CommandRLS("./data")
  grid.write_owned(42, user_id=1, tokens=[...])    # record owned by user 1
  grid.read(42, user_id=1)  # ✓
  grid.read(42, user_id=2)  # ✗ PermissionDenied
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
        """Read with RLS check. Parses commands from the token stream."""
        rec = super().read(record_id)
        if not rec or rec.is_tombstone:
            return None

        # Parse commands from the record's token stream
        owner = self._get_owner_from_tokens(rec.tokens)
        if owner is not None and owner != user_id and self._bypass_user is None:
            raise PermissionDenied(
                f"CommandRLS: user {user_id} cannot read record {record_id} (owner={owner})")
        return rec

    def _get_owner_from_tokens(self, tokens: List[Token]) -> Optional[int]:
        """Parse SPECIAL3 commands to find the AUTH owner."""
        parser = Parser()
        for t in tokens:
            parser.feed(t)
        parser.finalize()
        # Look for command tokens in output
        for p in parser.output:
            if isinstance(p, dict) and p.get('type') == 'command':
                if p.get('cmd') == 'AUTH':
                    # Find the next NUM after the command
                    # The argument comes after the command in the token stream
                    cmd_idx = parser.output.index(p)
                    for q in parser.output[cmd_idx:]:
                        if isinstance(q, ParsedNumber):
                            return q.value
        return None

    def delete(self, record_id: int, user_id: int = 0) -> bool:
        self.read(record_id, user_id)  # Verify ownership
        return super().delete(record_id)

    def as_admin(self):
        return _AC(self)


class _AC:
    def __init__(self, e): self.e = e
    def __enter__(self): self.e._bypass_user = -1; return self.e
    def __exit__(self, *a): self.e._bypass_user = None
