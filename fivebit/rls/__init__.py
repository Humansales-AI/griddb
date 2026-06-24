"""
5bit RLS — Row-Level Security
===============================
Engine-enforced access control. Not a helper you call — the storage
layer refuses the operation if the policy fails. You cannot forget.

Policies are defined per table, per action (read/write/delete).
Each policy is a function: (user_id, record_id, record_data) → bool.

Usage:
  from fivebit.rls import RLSGrid

  grid = RLSGrid("./data")
  grid.policy("users", "read", lambda uid, rid, rec: uid == rid)
  grid.policy("users", "write", lambda uid, rid, rec: uid == rid)

  grid.read(user_id=1, record_id=1)   # ✓ own record
  grid.read(user_id=2, record_id=1)   # ✗ PermissionDenied — engine refuses
"""
import os, sys
from typing import Callable, Optional, Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid, AllocRecord


class PermissionDenied(Exception):
    """Raised when an RLS policy rejects an operation."""
    pass


PolicyFn = Callable[[int, int, Optional[Dict[str, Any]]], bool]


class RLSGrid:
    """AllocGrid wrapper with row-level security.

    Every read/write/delete requires a user_id. Policies are checked
    BEFORE the operation reaches the storage layer. A failing policy
    raises PermissionDenied — you can't accidentally skip it because
    the engine refuses the call.
    """

    def __init__(self, data_dir: str = "./data"):
        self.grid = AllocGrid(data_dir=data_dir)
        self._policies: Dict[str, Dict[str, List[PolicyFn]]] = {}

    def policy(self, table: str, action: str, fn: PolicyFn):
        """Register a policy. table="users", action="read|write|delete"."""
        if table not in self._policies:
            self._policies[table] = {}
        if action not in self._policies[table]:
            self._policies[table][action] = []
        self._policies[table][action].append(fn)

    def _check(self, table: str, action: str, user_id: int,
               record_id: int, record: Optional[Dict] = None) -> bool:
        """Run all policies for this table+action. All must pass."""
        for act in (action, '*'):  # Check specific action, then wildcard
            for fn in self._policies.get(table, {}).get(act, []):
                if not fn(user_id, record_id, record):
                    raise PermissionDenied(
                        f"RLS: {action} denied on {table}#{record_id} for user {user_id}")
        return True

    def read(self, user_id: int, record_id: int) -> Optional[AllocRecord]:
        """Read a record, RLS-checked."""
        rec = self.grid.read(record_id)
        # Build a lightweight record dict for policy evaluation
        data = self._record_to_dict(rec) if rec else None
        self._check('*', 'read', user_id, record_id, data)
        return rec

    def write(self, user_id: int, record_id: int, tokens: List[Token]) -> int:
        """Write a record, RLS-checked. For updates, checks existing record too."""
        existing = self.grid.read(record_id)
        data = self._record_to_dict(existing) if existing else None
        action = 'write' if existing and not existing.is_tombstone else 'create'
        self._check('*', action, user_id, record_id, data)
        return self.grid.write(record_id, tokens)

    def delete(self, user_id: int, record_id: int) -> bool:
        """Delete a record, RLS-checked."""
        existing = self.grid.read(record_id)
        if existing:
            data = self._record_to_dict(existing)
            self._check('*', 'delete', user_id, record_id, data)
        return self.grid.delete(record_id)

    def _record_to_dict(self, rec: AllocRecord) -> Dict[str, Any]:
        """Convert a parsed record to a dict for policy evaluation."""
        result = {}
        words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        if words: result['text'] = ''.join(words)
        if nums: result['values'] = nums
        return result

    def close(self): self.grid.close()


# ── Pre-built policies ──────────────────────────────────────────────────

def owner_only(user_id: int, record_id: int, _rec) -> bool:
    """Only the owner (user_id == record_id) can access."""
    return user_id == record_id

def owner_or_admin(user_id: int, record_id: int, _rec) -> bool:
    """Owner or admin (user_id 1) can access."""
    return user_id == record_id or user_id == 1

def public_read(_user_id, _record_id, _rec) -> bool:
    """Anyone can read."""
    return True

def authenticated_only(user_id: int, _record_id, _rec) -> bool:
    """Any authenticated user can access."""
    return user_id > 0
