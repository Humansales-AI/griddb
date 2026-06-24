"""
5bit Page Cache — LRU read cache for AllocGrid
================================================
Drops read latency from ~200µs (disk I/O) to ~2µs (memory hit)
for frequently accessed records. Write-through — always consistent.

Usage:
  from fivebit.cache import CachedGrid
  grid = CachedGrid("./data", cache_size=1000)
  grid.read(42)  # first hit: disk, ~200µs
  grid.read(42)  # cache hit: ~2µs
"""
import os, sys, time, threading
from collections import OrderedDict
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from griddb_alloc import AllocGrid, AllocRecord
from binary_grid_db import Token, pack_to_bytes
from typing import List


class CachedGrid:
    """LRU-cached AllocGrid. Read cache only — writes go through immediately."""

    def __init__(self, data_dir: str = "./data", cache_size: int = 1000):
        self.grid = AllocGrid(data_dir=data_dir)
        self.cache: OrderedDict = OrderedDict()
        self.max_size = cache_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def read(self, record_id: int) -> Optional[AllocRecord]:
        with self._lock:
            if record_id in self.cache:
                self.cache.move_to_end(record_id)
                self.hits += 1
                return self.cache[record_id]

        rec = self.grid.read(record_id)
        if rec is not None:
            with self._lock:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # evict LRU
                self.cache[record_id] = rec
                self.misses += 1
        return rec

    def write(self, record_id: int, tokens: List[Token]) -> int:
        result = self.grid.write(record_id, tokens)
        with self._lock:
            self.cache.pop(record_id, None)  # invalidate
        return result

    def delete(self, record_id: int) -> bool:
        result = self.grid.delete(record_id)
        with self._lock:
            self.cache.pop(record_id, None)
        return result

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def close(self): self.grid.close()

    # Passthrough
    @property
    def total_entries(self): return self.grid.total_entries
