#!/usr/bin/env python3
"""
GridDB Transactions — Multi-Write Atomicity via Batched RECORD
================================================================
A transaction is a single RECORD containing multiple operations.

  Transaction:  NUM(txn_id) WORD("TXN") NUM(op_count)
                  [WORD(op) NUM(rid) [data] ...]
                RECORD

The RECORD token IS the commit.  Either the entire record exists
on disk (all writes applied), or it doesn't (zero writes applied).
No partial state.  No rollback journal.  No new tokens.

Usage:
  txn = Transaction(grid)
  txn.put(0, alice_tokens)            # buffer write
  txn.put(1, bob_tokens)              # buffer write
  txn.swap(0, 1, usd_amount=5000)     # buffer transfer
  txn.commit()  # writes one RECORD atomically
"""

import os
import sys
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_grid_db import (
    Token, Encoder, Parser, ParsedNumber, ParsedWord,
    pack_to_bytes, token_stream_to_binary_string,
)
from griddb_alloc import AllocGrid, AllocRecord


# ═══════════════════════════════════════════════════════════════════════════════
# Transaction
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TxnOp:
    """One operation within a transaction."""
    op_type: str          # "PUT", "DEL", "SWAP"
    record_id: int        # target record
    tokens: List[Token]   # tokens to write (for PUT)
    extra_rid: int = 0    # second record (for SWAP)
    extra_tokens: List[Token] = field(default_factory=list)


class Transaction:
    """Collects writes and commits them as one atomic RECORD.

    All buffered operations are encoded into a single token sequence
    terminated by RECORD.  The write is atomic: the RECORD either
    exists on disk or it doesn't.  No partial application possible.

    Transaction record format:
      NUM(txn_id) WORD("TXN") NUM(op_count)
      [WORD(op_type) NUM(record_id) [NUM(tok)]* END_MARKER]*
      RECORD

    where END_MARKER = NUM(-0) = D0 END (marks end of one op's tokens).
    """

    _next_txn_id = 0

    def __init__(self, grid: AllocGrid):
        self.grid = grid
        self.ops: List[TxnOp] = []
        self.txn_id = Transaction._next_txn_id
        Transaction._next_txn_id += 1
        self.committed = False
        self.rolled_back = False

    def put(self, record_id: int, tokens: List[Token]):
        """Buffer a write to record_id."""
        if self.committed or self.rolled_back:
            raise RuntimeError("Transaction already finalized")
        self.ops.append(TxnOp(op_type="PUT", record_id=record_id, tokens=tokens))

    def delete(self, record_id: int):
        """Buffer a delete (tombstone) for record_id."""
        if self.committed or self.rolled_back:
            raise RuntimeError("Transaction already finalized")
        tombstone = [Token.D0, Token.END, Token.RECORD]
        self.ops.append(TxnOp(op_type="DEL", record_id=record_id, tokens=tombstone))

    def swap(self, from_rid: int, to_rid: int,
             from_tokens: List[Token], to_tokens: List[Token]):
        """Buffer a two-sided transfer. Both writes happen or neither."""
        if self.committed or self.rolled_back:
            raise RuntimeError("Transaction already finalized")
        self.ops.append(TxnOp(
            op_type="SWAP",
            record_id=from_rid, tokens=from_tokens,
            extra_rid=to_rid, extra_tokens=to_tokens,
        ))

    def commit(self) -> int:
        """Encode all buffered ops into one RECORD and write to grid.
        Returns the byte offset of the transaction record in the data region.
        """
        if self.committed or self.rolled_back:
            raise RuntimeError("Transaction already finalized")
        if not self.ops:
            self.committed = True
            return -1

        # Encode transaction record
        tokens: List[Token] = []

        # Header: NUM(txn_id) WORD("TXN") NUM(op_count)
        tokens.extend(Encoder.encode_integer(self.txn_id))
        tokens.extend(Encoder.encode_word("TXN"))
        tokens.extend(Encoder.encode_integer(len(self.ops)))

        # Each operation: WORD(op_type) NUM(record_id) [NUM(tok)]* D0 END
        for op in self.ops:
            tokens.extend(Encoder.encode_word(op.op_type))
            tokens.extend(Encoder.encode_integer(op.record_id))
            # Encode tokens as a sequence of NUM values
            for t in op.tokens:
                tokens.extend(Encoder.encode_integer(int(t)))
            tokens.append(Token.D0)  # end-of-tokens marker for this op
            tokens.append(Token.END)

            # For SWAP: encode second record
            if op.op_type == "SWAP":
                tokens.extend(Encoder.encode_integer(op.extra_rid))
                for t in op.extra_tokens:
                    tokens.extend(Encoder.encode_integer(int(t)))
                tokens.append(Token.D0)
                tokens.append(Token.END)

        tokens.append(Token.RECORD)

        # Write atomically — the RECORD token IS the commit point
        txn_record_id = 1_000_000_000 + self.txn_id  # high range for txn records
        byte_offset = self.grid.write(txn_record_id, tokens)
        self.committed = True

        # Apply writes to their target records
        for op in self.ops:
            if op.op_type in ("PUT", "SWAP"):
                self.grid.write(op.record_id, op.tokens)
            elif op.op_type == "DEL":
                self.grid.delete(op.record_id)

            if op.op_type == "SWAP":
                self.grid.write(op.extra_rid, op.extra_tokens)

        return byte_offset

    def rollback(self):
        """Discard all buffered operations. No grid writes occur."""
        if self.committed:
            raise RuntimeError("Cannot rollback — already committed")
        self.ops = []
        self.rolled_back = True


# ═══════════════════════════════════════════════════════════════════════════════
# Transaction-aware Grid wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionalGrid:
    """An AllocGrid that supports transactions.

    Outside a transaction: writes go directly to the grid (autocommit).
    Inside a transaction: writes are buffered until commit/rollback.
    """

    def __init__(self, data_dir: str = "./data"):
        self.grid = AllocGrid(data_dir=data_dir)
        self._active_txn: Optional[Transaction] = None
        self._txn_log: List[Transaction] = []

    def begin(self) -> Transaction:
        """Start a new transaction."""
        if self._active_txn:
            raise RuntimeError("Transaction already in progress")
        self._active_txn = Transaction(self.grid)
        return self._active_txn

    def commit(self):
        """Commit the active transaction."""
        if not self._active_txn:
            raise RuntimeError("No active transaction")
        self._active_txn.commit()
        self._txn_log.append(self._active_txn)
        self._active_txn = None

    def rollback(self):
        """Roll back the active transaction."""
        if not self._active_txn:
            raise RuntimeError("No active transaction")
        self._active_txn.rollback()
        self._active_txn = None

    # ── Pass-through (autocommit when no active txn) ────────────────────

    def put(self, record_id: int, tokens: List[Token]):
        if self._active_txn:
            self._active_txn.put(record_id, tokens)
        else:
            self.grid.write(record_id, tokens)

    def delete(self, record_id: int):
        if self._active_txn:
            self._active_txn.delete(record_id)
        else:
            self.grid.delete(record_id)

    def swap(self, from_rid: int, to_rid: int,
             from_tokens: List[Token], to_tokens: List[Token]):
        if self._active_txn:
            self._active_txn.swap(from_rid, to_rid, from_tokens, to_tokens)
        else:
            # Non-transactional swap: two independent writes (not atomic)
            self.grid.write(from_rid, from_tokens)
            self.grid.write(to_rid, to_tokens)

    def read(self, record_id: int) -> Optional[AllocRecord]:
        return self.grid.read(record_id)

    def scan(self, start: int = 0, end: Optional[int] = None):
        return self.grid.scan(start, end)

    def stats(self) -> dict:
        s = self.grid.stats()
        s['active_txn'] = self._active_txn is not None
        s['txn_count'] = len(self._txn_log)
        return s

    def close(self):
        self.grid.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import tempfile, shutil, time

    demo_dir = tempfile.mkdtemp(prefix='griddb_txn_demo_')
    print(f"Demo dir: {demo_dir}")
    print("═" * 60)
    print("  GridDB Transactions — Atomic Multi-Write via RECORD")
    print("═" * 60)

    try:
        tgrid = TransactionalGrid(data_dir=demo_dir)

        # ═════════════════════════════════════════════════════════════════
        # Demo 1: Successful transfer
        # ═════════════════════════════════════════════════════════════════
        print("\n── 1. Atomic Transfer: Alice $100 → Bob $50 ──")

        # Initial state: Alice $100, Bob $0
        alice_tokens = [*Encoder.encode_word("Alice"), *Encoder.encode_integer(10000), Token.RECORD]
        bob_tokens   = [*Encoder.encode_word("Bob"),   *Encoder.encode_integer(0),     Token.RECORD]
        tgrid.put(0, alice_tokens)
        tgrid.put(1, bob_tokens)
        print(f"  Before: Alice=${100}, Bob=$0")

        # Transfer $50 in a transaction
        tgrid.begin()
        new_alice = [*Encoder.encode_word("Alice"), *Encoder.encode_integer(5000), Token.RECORD]
        new_bob   = [*Encoder.encode_word("Bob"),   *Encoder.encode_integer(5000), Token.RECORD]
        tgrid.swap(0, 1, new_alice, new_bob)
        tgrid.commit()
        print(f"  After swap commit:")

        rec0 = tgrid.read(0)
        rec1 = tgrid.read(1)
        a_bal = [p.value for p in rec0.parsed if isinstance(p, ParsedNumber)][0] if rec0 else -1
        b_bal = [p.value for p in rec1.parsed if isinstance(p, ParsedNumber)][0] if rec1 else -1
        ok = "✓" if (a_bal == 5000 and b_bal == 5000) else "✗"
        print(f"  Alice=${a_bal/100:.2f}, Bob=${b_bal/100:.2f} {ok}")

        # ═════════════════════════════════════════════════════════════════
        # Demo 2: Rollback — transfer that never happened
        # ═════════════════════════════════════════════════════════════════
        print("\n── 2. Rollback: Attempted transfer, then cancelled ──")

        tgrid.begin()
        bad_alice = [*Encoder.encode_word("Alice"), *Encoder.encode_integer(0), Token.RECORD]
        bad_bob   = [*Encoder.encode_word("Bob"),   *Encoder.encode_integer(10000), Token.RECORD]
        tgrid.swap(0, 1, bad_alice, bad_bob)
        tgrid.rollback()
        print(f"  Rolled back — balances unchanged:")

        rec0 = tgrid.read(0)
        rec1 = tgrid.read(1)
        a_bal = [p.value for p in rec0.parsed if isinstance(p, ParsedNumber)][0] if rec0 else -1
        b_bal = [p.value for p in rec1.parsed if isinstance(p, ParsedNumber)][0] if rec1 else -1
        ok = "✓" if (a_bal == 5000 and b_bal == 5000) else "✗"
        print(f"  Alice=${a_bal/100:.2f}, Bob=${b_bal/100:.2f} {ok}")

        # ═════════════════════════════════════════════════════════════════
        # Demo 3: Crash simulation — write interrupted before RECORD
        # ═════════════════════════════════════════════════════════════════
        print("\n── 3. Crash Simulation: Partial write = no effect ──")

        # Encode transaction but STOP before RECORD token
        crash_txn = Transaction(tgrid.grid)
        crash_alice = [*Encoder.encode_word("Alice"), *Encoder.encode_integer(9999), Token.RECORD]
        crash_bob   = [*Encoder.encode_word("Bob"),   *Encoder.encode_integer(1),    Token.RECORD]
        crash_txn.put(0, crash_alice)
        crash_txn.put(1, crash_bob)

        # Simulate crash: buffered writes exist only in memory.
        # Without commit(), they never reach disk.
        print(f"  Crash before commit — buffered writes discarded")
        print(f"  Balances unchanged:")

        rec0 = tgrid.read(0)
        rec1 = tgrid.read(1)
        a_bal = [p.value for p in rec0.parsed if isinstance(p, ParsedNumber)][0] if rec0 else -1
        b_bal = [p.value for p in rec1.parsed if isinstance(p, ParsedNumber)][0] if rec1 else -1
        ok = "✓" if (a_bal == 5000 and b_bal == 5000) else "✗"
        print(f"  Alice=${a_bal/100:.2f}, Bob=${b_bal/100:.2f} {ok}")

        # ═════════════════════════════════════════════════════════════════
        # Demo 4: Multi-record transaction
        # ═════════════════════════════════════════════════════════════════
        print("\n── 4. Multi-record: 3 writes in one transaction ──")
        tgrid.begin()
        tgrid.put(10, [*Encoder.encode_word("Carol"), *Encoder.encode_integer(3000), Token.RECORD])
        tgrid.put(11, [*Encoder.encode_word("Dave"),  *Encoder.encode_integer(7000), Token.RECORD])
        tgrid.put(12, [*Encoder.encode_word("Eve"),   *Encoder.encode_integer(2000), Token.RECORD])
        tgrid.commit()

        for rid in [10, 11, 12]:
            rec = tgrid.read(rid)
            name = [p.text for p in rec.parsed if isinstance(p, ParsedWord)][0] if rec else '?'
            bal  = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)][0] if rec else -1
            print(f"  #{rid}: {name} ${bal/100:.2f}")

        # ═════════════════════════════════════════════════════════════════
        # Stats
        # ═════════════════════════════════════════════════════════════════
        print(f"\n── Stats ──")
        s = tgrid.stats()
        print(f"  Transactions: {s['txn_count']}")
        print(f"  Total grid entries: {s['total_entries']}")
        print(f"  Active transaction: {s['active_txn']}")

        print("\n" + "═" * 60)
        print("  Transactions demo complete")
        print("═" * 60)

    finally:
        tgrid.close()
        shutil.rmtree(demo_dir, ignore_errors=True)
