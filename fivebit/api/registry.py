"""
5bit Table Registry — Collision-Free Namespacing
==================================================
Assigns each table a unique sequential namespace ID on first use.
Stored in the grid. No birthday collisions. No overflow.

Usage:
  reg = TableRegistry(grid)
  base = reg.base("users")     # → 0 * 10M (first table)
  base = reg.base("orders")    # → 1 * 10M (second table)
  base = reg.base("users")     # → 0 * 10M (same as before)
"""
import os, sys
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid

REGISTRY_BASE = 100_000  # Registry records live here
TABLE_STRIDE = 10_000_000


class TableRegistry:
    """Collision-free table namespace registry. Stored in the grid."""

    def __init__(self, grid: AllocGrid):
        self.grid = grid
        self._cache: Dict[str, int] = {}
        self._load()

    def _load(self):
        """Load existing mappings from grid."""
        for rid in range(REGISTRY_BASE, REGISTRY_BASE + 1000):
            rec = self.grid.read(rid)
            if not rec or rec.is_tombstone:
                continue
            words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
            nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
            if words and nums:
                self._cache[words[0]] = nums[0]

    def base(self, table_name: str) -> int:
        """Get the record ID base for a table. Assigns new namespace on first use."""
        if table_name in self._cache:
            return self._cache[table_name] * TABLE_STRIDE

        # Assign next sequential ID
        ns_id = len(self._cache)
        rid = REGISTRY_BASE + ns_id
        self.grid.write(rid, [
            *Encoder.encode_word(table_name),
            *Encoder.encode_integer(ns_id),
            Token.RECORD,
        ])
        self._cache[table_name] = ns_id
        return ns_id * TABLE_STRIDE

    def rid(self, table_name: str, local_id: int) -> int:
        """Get the global record ID for a table+local_id pair."""
        return self.base(table_name) + local_id
