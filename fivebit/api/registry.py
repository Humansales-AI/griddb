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
    """Collision-free table namespace registry. Separate records per field."""

    def __init__(self, grid: AllocGrid):
        self.grid = grid
        self._cache: Dict[str, int] = {}
        self._load()

    def _load(self):
        """Load mappings from per-field records (immune to fragmentation)."""
        ns_id = 0
        while True:
            name_rec = self.grid.read(REGISTRY_BASE + ns_id * 2)
            if not name_rec or name_rec.is_tombstone: break
            name = self.grid.reconstruct_all(name_rec.parsed)
            id_rec = self.grid.read(REGISTRY_BASE + ns_id * 2 + 1)
            if id_rec and not id_rec.is_tombstone:
                rid_nums = [p.value for p in id_rec.parsed if isinstance(p, ParsedNumber)]
                if rid_nums:
                    self._cache[name] = rid_nums[0]
            ns_id += 1

    def base(self, table_name: str) -> int:
        """Get the record ID base for a table. Assigns new namespace on first use."""
        if table_name in self._cache:
            return self._cache[table_name] * TABLE_STRIDE

        ns_id = len(self._cache)
        # Store name and ns_id in SEPARATE records — no fragmentation
        self.grid.write(REGISTRY_BASE + ns_id * 2, [
            *Encoder.encode_word(table_name), Token.RECORD,
        ])
        self.grid.write(REGISTRY_BASE + ns_id * 2 + 1, [
            *Encoder.encode_integer(ns_id), Token.RECORD,
        ])
        self._cache[table_name] = ns_id
        return ns_id * TABLE_STRIDE

    def rid(self, table_name: str, local_id: int) -> int:
        """Get the global record ID for a table+local_id pair."""
        return self.base(table_name) + local_id
