#!/usr/bin/env python3
"""
GridDB Positioned Grid — True O(1) Bit-Addressed Storage
==========================================================
Every record lives at a known position: record_id × STRIDE bits.
No scan, no index, no hash map. The address IS the identity.

write(42, tokens) → seek(42 × 1024) → write tokens → O(1)
read(42)          → seek(42 × 1024) → read until RECORD → O(1)

Architecture:
  main.grid      — fixed-stride grid file (pre-allocated in chunks)
  wal.grid       — WAL with SHA-256 chain (crash recovery)
  checkpoint.seq — last checkpointed WAL sequence

Tombstones: writing Token.D0 Token.END Token.RECORD marks a row as deleted.
"""

import os
import struct
import hashlib
import fcntl
from typing import List, Optional, Tuple
from dataclasses import dataclass

from binary_grid_db import (
    Token, Encoder, Parser, ParsedNumber, ParsedWord,
    pack_to_bytes, unpack_from_bytes,
    token_stream_to_binary_string,
)

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_STRIDE_BITS = 1024   # 128 bytes per row
GRID_MAGIC = 0x47524450      # "GRDP"
HEADER_FMT = ">III"          # magic, stride_bits, total_rows
HEADER_SIZE = struct.calcsize(HEADER_FMT)


@dataclass
class PositionedRecord:
    """A record read from a specific grid position."""
    record_id: int
    bit_offset: int
    tokens: List[Token]
    parsed: List  # ParsedNumber, ParsedWord, etc.
    is_tombstone: bool


# ═══════════════════════════════════════════════════════════════════════════════
# Positioned Grid
# ═══════════════════════════════════════════════════════════════════════════════

class PositionedGrid:
    """Fixed-stride, bit-addressable grid with O(1) read/write.

    Record N lives at bit offset N × stride_bits.
    Each row can hold up to stride_bits of token data.
    Records are self-delimiting (end with RECORD) so no length prefix needed.
    """

    def __init__(self, data_dir: str = "./data", stride_bits: int = DEFAULT_STRIDE_BITS):
        self.data_dir = data_dir
        self.stride_bits = stride_bits
        self.stride_bytes = (stride_bits + 7) // 8

        os.makedirs(data_dir, exist_ok=True)

        self.grid_path = os.path.join(data_dir, "main.grid")
        self.wal_path = os.path.join(data_dir, "wal.grid")
        self.cp_path = os.path.join(data_dir, "checkpoint.seq")

        self._total_rows = 0
        self._lock_fd = None

        self._bootstrap()

    # ── Bootstrap ────────────────────────────────────────────────────────

    def _bootstrap(self):
        """Initialize or load existing grid."""
        if os.path.exists(self.grid_path):
            with open(self.grid_path, 'rb') as f:
                header = f.read(HEADER_SIZE)
                if len(header) == HEADER_SIZE:
                    magic, stride, rows = struct.unpack(HEADER_FMT, header)
                    if magic == GRID_MAGIC:
                        self.stride_bits = stride
                        self.stride_bytes = (stride + 7) // 8
                        self._total_rows = rows
        else:
            self._create_empty_grid()

    def _create_empty_grid(self):
        """Create a new empty grid file with header."""
        with open(self.grid_path, 'wb') as f:
            f.write(struct.pack(HEADER_FMT, GRID_MAGIC, self.stride_bits, 0))
            f.flush()
            os.fsync(f.fileno())
        self._total_rows = 0

    # ── Locking ──────────────────────────────────────────────────────────

    def _acquire(self):
        self._lock_fd = open(self.grid_path, 'r+b')
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

    def _release(self):
        if self._lock_fd:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None

    # ── Core O(1) operations ─────────────────────────────────────────────

    def write(self, record_id: int, tokens: List[Token]) -> int:
        """Write tokens at position record_id × stride_bits. O(1).

        Tokens must fit within one stride. Returns the bit offset written to.
        """
        bit_offset = record_id * self.stride_bits
        token_bits = len(tokens) * 5

        if token_bits > self.stride_bits:
            raise ValueError(
                f"Record {record_id}: {token_bits} bits exceeds stride {self.stride_bits} bits. "
                f"Use a larger stride or split the record."
            )

        self._acquire()
        try:
            # Pack tokens to bytes
            packed, pad_len = pack_to_bytes(tokens)
            packed_bytes = bytes(packed)

            # Ensure file has enough rows
            if record_id >= self._total_rows:
                self._total_rows = record_id + 1
                self._update_header()

            # Calculate byte offset in file
            byte_offset = HEADER_SIZE + record_id * self.stride_bytes

            # Write packed tokens at the correct position
            with open(self.grid_path, 'r+b') as f:
                f.seek(byte_offset)
                f.write(packed_bytes)
                # Zero out remaining stride space
                remaining = self.stride_bytes - len(packed_bytes)
                if remaining > 0:
                    f.write(b'\x00' * remaining)
                f.flush()
                os.fsync(f.fileno())

            return bit_offset
        finally:
            self._release()

    def read(self, record_id: int) -> Optional[PositionedRecord]:
        """Read record at position record_id × stride_bits. O(1).

        Reads tokens from the stride position until RECORD or end of stride.
        Returns None if the row is empty or a tombstone.
        """
        bit_offset = record_id * self.stride_bits

        if record_id >= self._total_rows:
            return None

        byte_offset = HEADER_SIZE + record_id * self.stride_bytes

        # Read stride bytes
        with open(self.grid_path, 'rb') as f:
            f.seek(byte_offset)
            raw = f.read(self.stride_bytes)

        if not raw or len(raw) == 0:
            return None

        # Unpack: try different pad lengths until we find valid tokens ending with RECORD
        tokens = self._unpack_stride(raw)
        if tokens is None or len(tokens) == 0:
            return None

        # Parse
        parser = Parser()
        for t in tokens:
            parser.feed(t)
        parser.finalize()

        # Check for tombstone: [D0, END, RECORD] = [00000, 11110, 11100]
        is_tombstone = (
            len(tokens) == 3 and
            tokens[0] == Token.D0 and
            tokens[1] == Token.END and
            tokens[2] == Token.RECORD
        )

        return PositionedRecord(
            record_id=record_id,
            bit_offset=bit_offset,
            tokens=tokens,
            parsed=parser.output,
            is_tombstone=is_tombstone,
        )

    def delete(self, record_id: int) -> bool:
        """Mark a record as deleted by writing a tombstone. O(1)."""
        tombstone = [Token.D0, Token.END, Token.RECORD]
        self.write(record_id, tombstone)
        return True

    def _unpack_stride(self, raw: bytes) -> Optional[List[Token]]:
        """Unpack tokens from stride bytes. Tries pad lengths 0-7."""
        # Trim trailing null bytes (padding)
        raw = raw.rstrip(b'\x00')
        if len(raw) == 0:
            return None

        # Try each pad length (0-7). The correct one produces tokens ending with RECORD.
        for pad_len in range(8):
            try:
                # Calculate how many complete tokens we can extract
                bits_available = len(raw) * 8 - pad_len
                if bits_available <= 0:
                    continue
                num_tokens = bits_available // 5
                if num_tokens == 0:
                    continue

                tokens = unpack_from_bytes(bytearray(raw), pad_len, num_tokens)
                if tokens and tokens[-1] == Token.RECORD:
                    return tokens
            except Exception:
                continue

        return None

    # ── Header management ────────────────────────────────────────────────

    def _update_header(self):
        """Update the grid header with current total_rows."""
        with open(self.grid_path, 'r+b') as f:
            f.seek(0)
            f.write(struct.pack(HEADER_FMT, GRID_MAGIC, self.stride_bits, self._total_rows))
            f.flush()
            os.fsync(f.fileno())

    @property
    def total_rows(self) -> int:
        return self._total_rows

    # ── Scan utilities ──────────────────────────────────────────────────

    def scan(self, start: int = 0, end: Optional[int] = None) -> List[PositionedRecord]:
        """Scan a range of rows. O(end - start)."""
        if end is None:
            end = self._total_rows
        results = []
        for rid in range(start, min(end, self._total_rows)):
            rec = self.read(rid)
            if rec is not None and not rec.is_tombstone:
                results.append(rec)
        return results

    def find_first(self, predicate) -> Optional[PositionedRecord]:
        """Find the first record matching a predicate. O(n) scan."""
        for rid in range(self._total_rows):
            rec = self.read(rid)
            if rec is not None and not rec.is_tombstone and predicate(rec):
                return rec
        return None

    def close(self):
        self._release()


# ═══════════════════════════════════════════════════════════════════════════════
# Positioned Grid + WAL (crash-safe writes)
# ═══════════════════════════════════════════════════════════════════════════════

class PositionedGridWAL:
    """PositionedGrid with Write-Ahead Log for crash safety.

    Every write goes to WAL first (with SHA-256 chaining), then to the
    positioned grid. On recovery, un-checkpointed WAL entries are replayed.

    WAL entry format (extended):
      [magic "GPWL" 4B] [seq 4B] [record_id 4B] [token_count 4B]
      [prev_hash_offset 4B] [tokens... packed] [pad_len 4B] [SHA-256 32B]
    """

    WAL_MAGIC = 0x4750574C   # "GPWL"
    WAL_HDR_FMT = ">IIII"    # magic, seq, record_id, token_count
    WAL_PREV_FMT = ">i"      # prev_hash_offset
    WAL_PAD_FMT = ">I"       # pad_length

    WAL_HDR_SIZE = struct.calcsize(WAL_HDR_FMT)
    WAL_PREV_SIZE = struct.calcsize(WAL_PREV_FMT)
    WAL_PAD_SIZE = struct.calcsize(WAL_PAD_FMT)
    WAL_HASH_SIZE = 32
    WAL_OVERHEAD = WAL_HDR_SIZE + WAL_PREV_SIZE + WAL_PAD_SIZE + WAL_HASH_SIZE

    def __init__(self, data_dir: str = "./data", stride_bits: int = DEFAULT_STRIDE_BITS):
        self.grid = PositionedGrid(data_dir, stride_bits)
        self.wal_path = os.path.join(data_dir, "pos_wal.grid")

        self._next_seq = 0
        self._last_hash_offset = -1
        self._wal_entries = []  # in-memory cache of WAL entries

        if os.path.exists(self.wal_path):
            self._replay_wal()
        else:
            open(self.wal_path, 'wb').close()

    def _replay_wal(self):
        """Replay WAL entries into positioned grid."""
        entries = self._read_wal()
        for entry in entries:
            self.grid.write(entry['record_id'], entry['tokens'])
        self._wal_entries = entries
        if entries:
            self._next_seq = entries[-1]['seq'] + 1

    def _read_wal(self) -> List[dict]:
        """Read all entries from the positional WAL."""
        entries = []
        if not os.path.exists(self.wal_path):
            return entries

        with open(self.wal_path, 'rb') as f:
            data = f.read()

        offset = 0
        while offset + self.WAL_OVERHEAD <= len(data):
            magic, seq, record_id, token_count = struct.unpack_from(self.WAL_HDR_FMT, data, offset)
            if magic != self.WAL_MAGIC:
                break
            offset += self.WAL_HDR_SIZE

            prev_hash_offset = struct.unpack_from(self.WAL_PREV_FMT, data, offset)[0]
            offset += self.WAL_PREV_SIZE

            token_bits = token_count * 5
            token_bytes = (token_bits + 7) // 8
            if offset + token_bytes > len(data):
                break
            token_data = data[offset:offset + token_bytes]
            offset += token_bytes

            if offset + self.WAL_PAD_SIZE > len(data):
                break
            pad_len = struct.unpack_from(self.WAL_PAD_FMT, data, offset)[0]
            offset += self.WAL_PAD_SIZE

            if offset + self.WAL_HASH_SIZE > len(data):
                break
            stored_hash = data[offset:offset + self.WAL_HASH_SIZE]
            offset += self.WAL_HASH_SIZE

            tokens = unpack_from_bytes(bytearray(token_data), pad_len)

            entries.append({
                'seq': seq,
                'record_id': record_id,
                'tokens': tokens,
                'prev_hash_offset': prev_hash_offset,
                'sha256': stored_hash,
            })

        return entries

    def write(self, record_id: int, tokens: List[Token]):
        """Crash-safe write: WAL first, then grid."""
        self.grid._acquire()
        try:
            seq = self._next_seq
            self._next_seq += 1

            packed, pad_len = pack_to_bytes(tokens)

            # Build WAL entry
            header = struct.pack(self.WAL_HDR_FMT, self.WAL_MAGIC, seq, record_id, len(tokens))
            prev_bytes = struct.pack(self.WAL_PREV_FMT, self._last_hash_offset)
            pad_bytes = struct.pack(self.WAL_PAD_FMT, pad_len)

            content = header + prev_bytes + bytes(packed) + pad_bytes
            entry_hash = hashlib.sha256(content).digest()

            # Write WAL
            with open(self.wal_path, 'ab') as f:
                f.write(content + entry_hash)
                f.flush()
                os.fsync(f.fileno())

            # Update hash chain pointer
            self._last_hash_offset = os.path.getsize(self.wal_path) - self.WAL_HASH_SIZE

            # Write to positioned grid
            self.grid.write(record_id, tokens)

            self._wal_entries.append({
                'seq': seq, 'record_id': record_id, 'tokens': tokens,
                'prev_hash_offset': self._last_hash_offset, 'sha256': entry_hash,
            })
        finally:
            self.grid._release()

    def read(self, record_id: int) -> Optional[PositionedRecord]:
        return self.grid.read(record_id)

    def scan(self, start: int = 0, end: Optional[int] = None):
        return self.grid.scan(start, end)

    def close(self):
        self.grid.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import tempfile, shutil

    demo_dir = tempfile.mkdtemp(prefix='griddb_pos_demo_')
    print(f"Demo dir: {demo_dir}")
    print("═" * 55)
    print("  Positioned Grid — O(1) Bit-Addressed Storage")
    print("═" * 55)

    try:
        # 1. Create positioned grid
        print("\n── 1. Create Grid (stride = 512 bits = 64 bytes) ──")
        pg = PositionedGrid(data_dir=demo_dir, stride_bits=512)
        print(f"  Stride: {pg.stride_bits} bits = {pg.stride_bytes} bytes/row")
        print(f"  File: {pg.grid_path}")

        # 2. Write records at specific positions
        print("\n── 2. Write records at known positions ──")
        users = [
            (0, "Alice", 5000, 3000),    # id=0: row 0
            (1, "Bob", 10000, 0),         # id=1: row 1
            (42, "Charlie", 7500, 2000),  # id=42: row 42 (sparse!)
            (99, "Diana", 0, 8000),       # id=99: row 99
        ]

        for uid, name, usd, eur in users:
            tokens = [
                *Encoder.encode_word(name),
                *Encoder.encode_integer(usd),
                *Encoder.encode_integer(eur),
                Token.RECORD,
            ]
            bit_offset = pg.write(uid, tokens)
            print(f"  User #{uid:>3} '{name}' → bit offset {bit_offset} "
                  f"({len(tokens)} tokens, {len(tokens)*5} bits)")

        # 3. O(1) read
        print("\n── 3. O(1) Reads — seek directly to position ──")
        for uid in [0, 42, 99, 1, 7]:
            rec = pg.read(uid)
            if rec:
                vals = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
                names = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
                status = "☠ tombstone" if rec.is_tombstone else f"name={names[0] if names else '?'}, bal=[{vals}]"
                print(f"  read(#{uid:>3}) → bit {rec.bit_offset}: {status}")
            else:
                print(f"  read(#{uid:>3}) → empty row (no data)")

        # 4. Delete
        print("\n── 4. Delete user #1 (write tombstone) ──")
        pg.delete(1)
        rec = pg.read(1)
        print(f"  read(1) → tombstone: {rec.is_tombstone if rec else 'N/A'}")

        # 5. Scan range
        print("\n── 5. Scan rows 0-50 ──")
        results = pg.scan(0, 50)
        for r in results:
            names = [p.text for p in r.parsed if isinstance(p, ParsedWord)]
            vals = [p.value for p in r.parsed if isinstance(p, ParsedNumber)]
            print(f"  #{r.record_id:>3}: {names[0] if names else '?'} {vals}")

        # 6. WAL-integrated grid
        print("\n── 6. PositionedGridWAL (crash-safe writes) ──")
        wal_grid = PositionedGridWAL(data_dir=demo_dir + "_wal", stride_bits=512)
        wal_grid.write(0, Encoder.encode_record("Alice-WAL", 9999, 1111))
        wal_grid.write(50, Encoder.encode_record("Bob-WAL", 5555, 2222))

        rec0 = wal_grid.read(0)
        rec50 = wal_grid.read(50)
        if rec0:
            names = [p.text for p in rec0.parsed if isinstance(p, ParsedWord)]
            print(f"  WAL write → read(0): {names}")
        if rec50:
            names = [p.text for p in rec50.parsed if isinstance(p, ParsedWord)]
            print(f"  WAL write → read(50): {names}")

        wal_grid.close()

        # 7. Verify O(1) property
        print("\n── 7. O(1) Verification ──")
        import time
        # Write 1000 records
        for i in range(1000):
            pg.write(i, Encoder.encode_record(f"USER-{i}", i * 100, i * 50))

        # Time reads at different positions
        positions = [0, 500, 999, 42, 777]
        for pos in positions:
            start = time.perf_counter()
            rec = pg.read(pos)
            elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds
            names = [p.text for p in rec.parsed if isinstance(p, ParsedWord)] if rec else ['?']
            print(f"  read(#{pos:>3}) → {names[0]:<10} in {elapsed:.1f}µs")

        print("\n" + "═" * 55)
        print("  Positioned Grid demo complete")
        print("═" * 55)

    finally:
        pg.close()
        shutil.rmtree(demo_dir + "_wal", ignore_errors=True)
        shutil.rmtree(demo_dir)
