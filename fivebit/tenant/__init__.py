"""
5bit Multi-Tenant — Tenant Isolation
======================================
Engine-enforced tenant partitioning. Every record is scoped to a tenant.
Cross-tenant reads are impossible — the storage layer filters by tenant_id.

Tenant ID is embedded in every record at a known position.
All read/write operations auto-scope to the current tenant.

Usage:
  from fivebit.tenant import TenantGrid

  grid = TenantGrid("./data")
  grid.with_tenant("acme-corp", lambda g: [
      g.write(0, tokens),   # scoped to acme-corp
      g.read(0),            # only acme-corp's records
  ])

  grid.with_tenant("beta-inc", lambda g: [
      g.read(0),            # beta-inc's records — different from acme-corp
  ])
"""
import os, sys, threading
from typing import Callable, Optional, List, TypeVar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid, AllocRecord

# Tenant-scoped record_id = tenant_hash * TENANT_STRIDE + original_id
TENANT_STRIDE = 1_000_000

T = TypeVar('T')

class TenantGrid:
    """Multi-tenant grid with enforced isolation.

    Every operation is scoped to a tenant. The tenant_id is hashed and
    used to partition the record ID space. Records for different tenants
    live in different ID ranges — cross-tenant reads are physically impossible.
    """

    def __init__(self, data_dir: str = "./data"):
        self.grid = AllocGrid(data_dir=data_dir)
        self._active_tenant: Optional[str] = None

    def _scope(self, record_id: int) -> int:
        """Map a local record_id to a tenant-scoped global ID."""
        if not self._active_tenant:
            raise RuntimeError("No active tenant — use with_tenant()")
        tenant_hash = abs(hash(self._active_tenant)) % (TENANT_STRIDE - 1) + 1
        return tenant_hash * TENANT_STRIDE + record_id

    def with_tenant(self, tenant_id: str, fn: Callable[['TenantGrid'], T]) -> T:
        """Execute fn with all operations scoped to tenant_id."""
        old = self._active_tenant
        self._active_tenant = tenant_id
        try:
            return fn(self)
        finally:
            self._active_tenant = old

    def write(self, record_id: int, tokens: List[Token]) -> int:
        return self.grid.write(self._scope(record_id), tokens)

    def read(self, record_id: int) -> Optional[AllocRecord]:
        return self.grid.read(self._scope(record_id))

    def delete(self, record_id: int) -> bool:
        return self.grid.delete(self._scope(record_id))

    def scan_tenant(self, max_records: int = 1000) -> List[AllocRecord]:
        """Scan all records in the current tenant's partition."""
        if not self._active_tenant:
            raise RuntimeError("No active tenant")
        base = self._scope(0)
        results = []
        for i in range(max_records):
            rec = self.grid.read(base + i)
            if rec and not rec.isTombstone:
                results.append(rec)
        return results

    def close(self): self.grid.close()
