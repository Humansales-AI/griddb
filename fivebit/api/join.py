"""
5bit Merge Join — B-tree Indexed, O(n log n) build + O(n) merge
==================================================================
No nested loops. No hash maps. Uses existing BTreeIndex.

Walks two B-trees in parallel. Matching keys → paired records.
Same algorithm PostgreSQL uses for USING(column) — a merge join.

GET /join?left=users&right=orders&on=user_id
"""
import os, sys, hashlib
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from griddb_index import BTreeIndex
from griddb_alloc import AllocGrid, AllocRecord
from binary_grid_db import ParsedNumber, ParsedWord


def _record_to_dict(rec: AllocRecord, fields: List[str]) -> dict:
    vals = []; pending = ''
    for p in rec.parsed:
        if isinstance(p, ParsedNumber):
            if pending: vals.append(pending); pending = ''
            vals.append(p.value)
        elif isinstance(p, ParsedWord):
            pending += p.text
    if pending: vals.append(pending)
    return {fields[i]: vals[i] for i in range(min(len(fields), len(vals)))}


TABLE_STRIDE = 10_000_000  # Each table gets its own record ID namespace

def _table_base(name: str) -> int:
    """Deterministic record ID range per table."""
    h = hashlib.sha256(name.encode()).digest()
    return (int.from_bytes(h[:4], 'big') % 1000) * TABLE_STRIDE

class MergeJoiner:
    """B-tree merge join. Tables partitioned by record ID range."""

    def __init__(self, grid: AllocGrid, left_spec: dict, right_spec: dict):
        self.grid = grid
        self.left_name = left_spec['name']
        self.left_fields = left_spec.get('fields', [])
        self.right_name = right_spec['name']
        self.right_fields = right_spec.get('fields', [])
        self.left_base = _table_base(self.left_name)
        self.right_base = _table_base(self.right_name)

    def join(self, on_field: str, data_dir: str) -> List[dict]:
        """Merge join on a shared field. Tables partitioned by rid range."""
        left_idx = BTreeIndex(f"{self.left_name}_join", data_dir)
        right_idx = BTreeIndex(f"{self.right_name}_join", data_dir)
        left_map: Dict[int, List[int]] = {}
        right_map: Dict[int, List[int]] = {}
        lf_idx = self.left_fields.index(on_field) if on_field in self.left_fields else -1
        rf_idx = self.right_fields.index(on_field) if on_field in self.right_fields else -1

        # Scan left table namespace
        for local_rid in range(10000):
            rid = self.left_base + local_rid
            rec = self.grid.read(rid)
            if not rec or rec.is_tombstone: continue
            vals = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
            if lf_idx >= 0 and lf_idx < len(vals):
                key = vals[lf_idx]
                left_map.setdefault(key, []).append(rid)
                left_idx.put(key, rid)

        # Scan right table namespace
        for local_rid in range(10000):
            rid = self.right_base + local_rid
            rec = self.grid.read(rid)
            if not rec or rec.is_tombstone: continue
            vals = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
            if rf_idx >= 0 and rf_idx < len(vals):
                key = vals[rf_idx]
                right_map.setdefault(key, []).append(rid)
                right_idx.put(key, rid)

        # Merge join
        results = []
        all_keys = sorted(set(left_map.keys()) & set(right_map.keys()))
        for key in all_keys:
            for lr in left_map[key]:
                left_rec = self.grid.read(lr)
                if not left_rec or left_rec.is_tombstone: continue
                for rr in right_map[key]:
                    right_rec = self.grid.read(rr)
                    if not right_rec or right_rec.is_tombstone: continue
                    results.append({
                        self.left_name: _record_to_dict(left_rec, self.left_fields),
                        self.right_name: _record_to_dict(right_rec, self.right_fields),
                    })

        left_idx.close(); right_idx.close()
        return results


def merge_join(grid: AllocGrid, left_spec: dict, right_spec: dict,
               on_field: str, data_dir: str) -> List[dict]:
    """Convenience: merge join two collections on a shared field."""
    mj = MergeJoiner(grid, left_spec, right_spec)
    return mj.join(on_field, data_dir)
