#!/usr/bin/env python3
"""
GridDB AllocGrid — Two-Level O(1) Storage at Any Scale
========================================================
Level 1: Allocation Table (fixed-stride, record_id → offset+length)
Level 2: Data Region (variable-length token blobs)

read(42):  alloc[42] → (offset, length) → data.seek(offset) → O(1)
write(42): append tokens to data region → update alloc[42] → O(1)
delete(42): mark alloc[42] flags = tombstone → O(1)

Alloc table: exactly 16 bytes per entry (offset:8, length:4, flags:4)
Entry N is at byte N × 16 → O(1) without any parsing

Scales to billions of records. Sparse by design — row 1,000,000
occupies disk only when data is written there.
"""

import os
import struct
import fcntl
import time
import hashlib
from typing import List, Optional, Tuple
from dataclasses import dataclass

from binary_grid_db import (
    Token, Encoder, Parser, ParsedNumber, ParsedWord,
    pack_to_bytes, unpack_from_bytes,
    token_stream_to_binary_string,
)

# ── Constants ──────────────────────────────────────────────────────────────

ALLOC_ENTRY_SIZE = 16                # bytes per alloc entry
ALLOC_ENTRY_FMT  = ">QII"           # offset(uint64), length(uint32), flags(uint32)
ALLOC_HEADER_FMT = ">II"            # magic, version
ALLOC_HEADER_SIZE = struct.calcsize(ALLOC_HEADER_FMT)
ALLOC_MAGIC = 0x414C4F43            # "ALOC"

# Flags
FLAG_FREE      = 0
FLAG_ALLOCATED = 1
FLAG_TOMBSTONE = 2

# Data region
DATA_MAGIC = 0x44415441            # "DATA"
DATA_HEADER_SIZE = 12               # [magic: uint32][data_end_offset: uint64]
DATA_HEADER_FMT  = ">IQ"


@dataclass
class AllocEntry:
    """One row in the allocation table."""
    record_id: int
    byte_offset: int       # byte offset in data region (0 = unallocated)
    bit_length: int        # length in bits
    flags: int             # 0=free, 1=allocated, 2=tombstone

    @property
    def byte_length(self) -> int:
        return (self.bit_length + 7) // 8

    @property
    def is_free(self) -> bool:
        return self.flags == FLAG_FREE or self.byte_offset == 0


@dataclass
class AllocRecord:
    """A full record read from the AllocGrid."""
    record_id: int
    tokens: List[Token]
    parsed: List           # ParsedNumber, ParsedWord, etc.
    byte_offset: int
    bit_length: int
    flags: int

    @property
    def is_tombstone(self) -> bool:
        return self.flags == FLAG_TOMBSTONE


# ═══════════════════════════════════════════════════════════════════════════════
# AllocGrid
# ═══════════════════════════════════════════════════════════════════════════════

class AllocGrid:
    """Two-level grid: allocation table + data region.

    Allocation table: file at alloc_path, 16 bytes/entry.
      Entry N at byte offset HEADER_SIZE + N × 16.
      Contains: (data_offset: uint64, bit_length: uint32, flags: uint32)

    Data region: file at data_path.
      Tokens packed to bytes, appended at end.
      Entry points to (offset, length) within this file.
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.alloc_path = os.path.join(data_dir, "alloc.grid")
        self.data_path = os.path.join(data_dir, "data.grid")

        self._lock_fd = None
        self._data_end = DATA_HEADER_SIZE  # next write position in data region

        self._bootstrap()

    # ── Bootstrap ────────────────────────────────────────────────────────

    def _bootstrap(self):
        """Initialize or load existing files."""
        if os.path.exists(self.alloc_path):
            with open(self.alloc_path, 'rb') as f:
                hdr = f.read(ALLOC_HEADER_SIZE)
                if len(hdr) == ALLOC_HEADER_SIZE:
                    magic, ver = struct.unpack(ALLOC_HEADER_FMT, hdr)
                    if magic != ALLOC_MAGIC:
                        raise RuntimeError(f"Invalid alloc file magic: {magic:08x}")
        else:
            self._create_alloc()

        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as f:
                end_bytes = f.read(DATA_HEADER_SIZE)
                if len(end_bytes) == DATA_HEADER_SIZE:
                    _, self._data_end = struct.unpack(DATA_HEADER_FMT, end_bytes)
        else:
            self._create_data()

    def _create_alloc(self):
        with open(self.alloc_path, 'wb') as f:
            f.write(struct.pack(ALLOC_HEADER_FMT, ALLOC_MAGIC, 1))
            f.flush(); os.fsync(f.fileno())

    def _create_data(self):
        with open(self.data_path, 'wb') as f:
            f.write(struct.pack(DATA_HEADER_FMT, DATA_MAGIC, DATA_HEADER_SIZE))
            f.flush(); os.fsync(f.fileno())
        self._data_end = DATA_HEADER_SIZE

    # ── Locking ──────────────────────────────────────────────────────────

    def _acquire(self):
        self._lock_fd = open(self.alloc_path, 'r+b')
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

    def _release(self):
        if self._lock_fd:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None

    # ── Alloc table operations ───────────────────────────────────────────

    def _read_alloc_entry(self, record_id: int) -> AllocEntry:
        """Read one entry from the allocation table. O(1)."""
        offset = ALLOC_HEADER_SIZE + record_id * ALLOC_ENTRY_SIZE

        with open(self.alloc_path, 'rb') as f:
            f.seek(offset)
            raw = f.read(ALLOC_ENTRY_SIZE)

        if len(raw) < ALLOC_ENTRY_SIZE:
            # Past end of file → free
            return AllocEntry(record_id=record_id, byte_offset=0, bit_length=0, flags=FLAG_FREE)

        data_off, bit_len, flags = struct.unpack(ALLOC_ENTRY_FMT, raw)
        return AllocEntry(record_id=record_id, byte_offset=data_off, bit_length=bit_len, flags=flags)

    def _write_alloc_entry(self, entry: AllocEntry):
        """Write one entry to the allocation table. O(1)."""
        alloc_offset = ALLOC_HEADER_SIZE + entry.record_id * ALLOC_ENTRY_SIZE

        # Ensure file is large enough
        needed_size = alloc_offset + ALLOC_ENTRY_SIZE
        with open(self.alloc_path, 'r+b') as f:
            f.seek(0, os.SEEK_END)
            current_size = f.tell()
            if current_size < needed_size:
                f.seek(needed_size - 1)
                f.write(b'\x00')
                f.flush()

        with open(self.alloc_path, 'r+b') as f:
            f.seek(alloc_offset)
            f.write(struct.pack(ALLOC_ENTRY_FMT, entry.byte_offset, entry.bit_length, entry.flags))
            f.flush()
            os.fsync(f.fileno())

    @property
    def total_entries(self) -> int:
        """Number of alloc table entries (allocated rows)."""
        with open(self.alloc_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
        if size <= ALLOC_HEADER_SIZE:
            return 0
        return (size - ALLOC_HEADER_SIZE) // ALLOC_ENTRY_SIZE

    # ── Core O(1) operations ─────────────────────────────────────────────

    def write(self, record_id: int, tokens: List[Token]) -> int:
        """Write tokens and update alloc table. O(1).

        Returns the byte offset in the data region.
        """
        self._acquire()
        try:
            # Pack tokens
            packed, pad_len = pack_to_bytes(tokens)
            packed_bytes = bytes(packed)
            bit_length = len(tokens) * 5

            # Append to data region — re-read _data_end from file to avoid stale cache
            with open(self.data_path, 'rb') as f:
                hdr = f.read(DATA_HEADER_SIZE)
                if len(hdr) == DATA_HEADER_SIZE:
                    _, self._data_end = struct.unpack(DATA_HEADER_FMT, hdr)
            data_offset = self._data_end
            with open(self.data_path, 'r+b') as f:
                f.seek(data_offset)
                f.write(packed_bytes)
                f.flush()
                os.fsync(f.fileno())

            # Update data_end
            self._data_end = data_offset + len(packed_bytes)
            with open(self.data_path, 'r+b') as f:
                f.seek(0)
                f.write(struct.pack(DATA_HEADER_FMT, DATA_MAGIC, self._data_end))
                f.flush()
                os.fsync(f.fileno())

            # Update alloc table
            entry = AllocEntry(
                record_id=record_id,
                byte_offset=data_offset,
                bit_length=bit_length,
                flags=FLAG_ALLOCATED,
            )
            self._write_alloc_entry(entry)

            return data_offset
        finally:
            self._release()

    def read(self, record_id: int) -> Optional[AllocRecord]:
        """Read a record via the alloc table. O(1)."""
        # Look up alloc entry
        entry = self._read_alloc_entry(record_id)
        if entry.is_free:
            return None

        # Read from data region
        byte_len = entry.byte_length
        with open(self.data_path, 'rb') as f:
            f.seek(entry.byte_offset)
            raw = f.read(byte_len)

        if not raw or len(raw) == 0:
            return None

        # Unpack tokens (try pad lengths 0-7)
        tokens = self._unpack(raw, entry.bit_length)
        if tokens is None:
            return None

        # Parse
        parser = Parser()
        for t in tokens:
            parser.feed(t)
        parser.finalize()

        return AllocRecord(
            record_id=record_id,
            tokens=tokens,
            parsed=parser.output,
            byte_offset=entry.byte_offset,
            bit_length=entry.bit_length,
            flags=entry.flags,
        )

    def delete(self, record_id: int) -> bool:
        """Mark record as tombstone in alloc table. O(1)."""
        entry = self._read_alloc_entry(record_id)
        if entry.is_free:
            return False

        entry.flags = FLAG_TOMBSTONE
        self._write_alloc_entry(entry)
        return True

    def _unpack(self, raw: bytes, expected_bit_length: int) -> Optional[List[Token]]:
        """Unpack bytes to tokens, trying pad lengths."""
        expected_tokens = expected_bit_length // 5
        for pad_len in range(8):
            try:
                tokens = unpack_from_bytes(bytearray(raw), pad_len, expected_tokens)
                if len(tokens) == expected_tokens:
                    return tokens
            except Exception:
                continue
        return None

    # ── Scan ─────────────────────────────────────────────────────────────

    def scan(self, start: int = 0, end: Optional[int] = None) -> List[AllocRecord]:
        """Scan a range of alloc entries, returning valid records."""
        if end is None:
            end = self.total_entries
        results = []
        for rid in range(start, end):
            rec = self.read(rid)
            if rec is not None and not rec.is_tombstone:
                results.append(rec)
        return results

    def occupied_rows(self) -> List[int]:
        """List all row IDs that have data (free or allocated, not empty)."""
        total = self.total_entries
        rows = []
        for rid in range(total):
            entry = self._read_alloc_entry(rid)
            if entry.flags != FLAG_FREE or entry.byte_offset != 0:
                rows.append(rid)
        return rows

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self.total_entries
        allocated = 0
        tombstones = 0
        total_data_bits = 0

        for rid in range(min(total, 100000)):  # Cap at 100k for stats
            entry = self._read_alloc_entry(rid)
            if entry.flags == FLAG_ALLOCATED:
                allocated += 1
                total_data_bits += entry.bit_length
            elif entry.flags == FLAG_TOMBSTONE:
                tombstones += 1

        return {
            'total_entries': total,
            'allocated': allocated,
            'tombstones': tombstones,
            'free': total - allocated - tombstones,
            'data_bits': total_data_bits,
            'data_bytes': total_data_bits // 8 + (1 if total_data_bits % 8 else 0),
            'alloc_file_bytes': os.path.getsize(self.alloc_path) if os.path.exists(self.alloc_path) else 0,
            'data_file_bytes': os.path.getsize(self.data_path) if os.path.exists(self.data_path) else 0,
        }

    def close(self):
        self._release()


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import tempfile, shutil

    demo_dir = tempfile.mkdtemp(prefix='griddb_alloc_demo_')
    print(f"Demo dir: {demo_dir}")
    print("═" * 60)
    print("  AllocGrid — Two-Level O(1) Storage at Any Scale")
    print("═" * 60)

    try:
        # 1. Create
        print("\n── 1. Create AllocGrid ──")
        ag = AllocGrid(data_dir=demo_dir)
        print(f"  Alloc table: {ag.alloc_path} ({ALLOC_ENTRY_SIZE} bytes/entry)")
        print(f"  Data region: {ag.data_path}")
        print(f"  Initial entries: {ag.total_entries}")

        # 2. Write records at specific IDs
        print("\n── 2. Write at specific record IDs ──")
        records = [
            (0, "Alice",   5000, 3000),
            (1, "Bob",    10000, 0),
            (1000000, "Charlie-sparse", 7500, 2000),  # sparse!
            (42, "Douglas", 0, 8000),
        ]

        for rid, name, usd, eur in records:
            tokens = [
                *Encoder.encode_word(name),
                *Encoder.encode_integer(usd),
                *Encoder.encode_integer(eur),
                Token.RECORD,
            ]
            data_offset = ag.write(rid, tokens)
            print(f"  write(#{rid:>7}) → data offset {data_offset:>6} "
                  f"({len(tokens)} tokens, {len(tokens)*5} bits)")

        # 3. O(1) reads
        print("\n── 3. O(1) Reads ──")
        for rid in [0, 1000000, 42, 1, 999]:
            rec = ag.read(rid)
            if rec:
                names = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
                vals = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
                status = "☠ tombstone" if rec.is_tombstone else f"name={names[0] if names else '?'}, vals={vals}"
                print(f"  read(#{rid:>7}) → offset {rec.byte_offset}: {status}")
            else:
                print(f"  read(#{rid:>7}) → empty (no data)")

        # 4. Delete
        print("\n── 4. Delete #1 ──")
        ag.delete(1)
        rec = ag.read(1)
        print(f"  read(1) → tombstone: {rec.is_tombstone if rec else 'N/A'}")

        # 5. Scan
        print("\n── 5. Scan all entries ──")
        rows = ag.occupied_rows()
        print(f"  Occupied rows: {rows}")
        for r in ag.scan(0, 100):
            names = [p.text for p in r.parsed if isinstance(p, ParsedWord)]
            vals = [p.value for p in r.parsed if isinstance(p, ParsedNumber)]
            print(f"  #{r.record_id:>7}: {names[0] if names else '?'} {vals}")

        # 6. Stats
        print("\n── 6. Stats ──")
        s = ag.stats()
        for k, v in s.items():
            print(f"  {k}: {v}")

        # 7. O(1) benchmark
        print("\n── 7. O(1) Benchmark (10,000 records) ──")
        for i in range(10000):
            ag.write(i, [Token.D1, Token.D2, Token.D3, Token.END, Token.RECORD])

        times = []
        for pos in [0, 5000, 9999, 42, 7777]:
            start = time.perf_counter()
            rec = ag.read(pos)
            elapsed = (time.perf_counter() - start) * 1_000_000
            times.append(elapsed)
            if rec:
                print(f"  read(#{pos:>5}) → {len(rec.tokens)} tokens in {elapsed:.1f}µs")

        avg = sum(times) / len(times)
        print(f"\n  Average read time: {avg:.1f}µs (O(1))")
        print(f"  Alloc file: {os.path.getsize(ag.alloc_path):,} bytes "
              f"({ag.total_entries:,} entries)")
        print(f"  Data file: {os.path.getsize(ag.data_path):,} bytes")

        print("\n" + "═" * 60)
        print("  AllocGrid demo complete")
        print("═" * 60)

    finally:
        ag.close()
        shutil.rmtree(demo_dir)
