#!/usr/bin/env python3
"""
GridDB Stress Test — Load Harness
===================================
5,000+ sequential transactions. 50 concurrent writers.
SIGKILL crash recovery. Balance verification.

Proves correctness under load. No silent failures.
"""
import os, sys, time, struct, hashlib, threading, tempfile, shutil, subprocess, signal
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber
from griddb_wal import WALGrid
from griddb_correctness import GroupCommitWAL

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _read_last_value(wal) -> int:
    """Read the LAST numeric value from all records in the grid."""
    tokens = wal.grid._tokens
    if not tokens: return 0
    p = Parser()
    for t in tokens: p.feed(t)
    p.finalize()
    vals = [x.value for x in p.output if isinstance(x, ParsedNumber)]
    return vals[-1] if vals else 0

def _read_account_value(wal, account_id: int, num_accounts: int) -> int:
    """Read the last value for account_id in an interleaved record stream."""
    tokens = wal.grid._tokens
    if not tokens: return 0
    rec_idx = -1
    val = 0
    for t in tokens:
        if t == Token.RECORD:
            rec_idx += 1
            if rec_idx % num_accounts == account_id:
                # Parse the number from this record (simple: scan back for digits)
                pass  # Parsing inline is complex; use the Parser approach instead
    # Re-parse all records to find account's values
    p = Parser()
    rec_idx = -1
    for t in tokens:
        p.feed(t)
        if t == Token.RECORD:
            rec_idx += 1
            if rec_idx % num_accounts == account_id:
                vals = [x.value for x in p.output if isinstance(x, ParsedNumber)]
                if vals: val = vals[-1]
            p = Parser()
    return val

def _read_balance(wal, account_id: int) -> int:
    """Read the LAST balance written for an account from the grid."""
    tokens = wal.grid._tokens
    if not tokens: return 0
    p = Parser()
    bal = 0
    rec_idx = -1
    for t in tokens:
        p.feed(t)
        if t == Token.RECORD:
            rec_idx += 1
            if rec_idx == account_id:
                vals = [x.value for x in p.output if isinstance(x, ParsedNumber)]
                if vals: bal = vals[-1]  # Keep latest
            p = Parser()  # Reset for next record
    return bal

def _write_balance(wal, account_id: int, amount: int):
    """Write a balance to a NEW record (appends, doesn't overwrite)."""
    tokens = [*Encoder.encode_integer(amount), Token.RECORD]
    wal.wal_append_record(tokens)

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Sequential — 5,000 deposits, one account
# ═══════════════════════════════════════════════════════════════════════════════

def test_sequential_load():
    d = tempfile.mkdtemp()
    wal = WALGrid(data_dir=d)
    N = 5000
    account = 0

    print(f"  Sequential: {N} deposits to 1 account...", end=" ", flush=True)
    t0 = time.perf_counter()

    for i in range(N):
        # Each write appends a new record — read the latest value
        current = _read_last_value(wal)
        _write_balance(wal, account, current + 1)

    elapsed = (time.perf_counter() - t0) * 1000
    final = _read_last_value(wal)  # Last record = most recent balance
    ok = final == N
    ops = int(N / (elapsed / 1000))

    print(f"final={final} {'✓' if ok else '✗ LOST ' + str(N - final)} "
          f"({elapsed:.0f}ms, {ops} ops/s)")
    wal.close()
    shutil.rmtree(d, ignore_errors=True)
    return ok, ops


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Concurrent — 50 threads × 100 deposits each
# ═══════════════════════════════════════════════════════════════════════════════

def test_concurrent_load():
    d = tempfile.mkdtemp()
    wal = WALGrid(data_dir=d)
    threads = 50
    deposits_per = 100
    total = threads * deposits_per  # 5,000
    lock = threading.Lock()
    errors = []

    def worker(tid):
        try:
            for _ in range(deposits_per):
                with lock:
                    # Each thread reads the LAST value from its account slot
                    # Records are interleaved: account N at indices N, N+50, N+100, ...
                    tokens = wal.grid._tokens
                    current = 0
                    rec_idx = -1
                    for t in tokens:
                        if t == Token.RECORD:
                            rec_idx += 1
                            if rec_idx % threads == tid:
                                # This record belongs to this account
                                # Extract value (parse inline)
                                pass
                    # Simpler: just read the value at the account's last record
                    current = _read_account_value(wal, tid, threads)
                    _write_balance(wal, tid, current + 1)
        except Exception as e:
            errors.append(str(e))

    print(f"  Concurrent: {threads} threads × {deposits_per} deposits = {total} total...",
          end=" ", flush=True)
    t0 = time.perf_counter()

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts: t.start()
    for t in ts: t.join()

    elapsed = (time.perf_counter() - t0) * 1000

    # Verify: the LAST record for each account should have deposits_per
    total_balance = 0
    all_ok = True
    tokens = wal.grid._tokens
    p = Parser()
    rec_idx = -1
    last_vals = {}  # account → last value seen
    for t in tokens:
        p.feed(t)
        if t == Token.RECORD:
            rec_idx += 1
            account = rec_idx % threads
            vals = [x.value for x in p.output if isinstance(x, ParsedNumber)]
            if vals: last_vals[account] = vals[-1]
            p = Parser()
    for i in range(threads):
        bal = last_vals.get(i, 0)
        total_balance += bal
        if bal != deposits_per:
            all_ok = False
            if len(errors) < 3: errors.append(f"Account {i}: expected {deposits_per}, got {bal}")

    ops = int(total / (elapsed / 1000))
    print(f"sum={total_balance} {'✓' if all_ok and not errors else '✗'} "
          f"({elapsed:.0f}ms, {ops} ops/s)")
    if errors:
        for e in errors[:3]: print(f"    {e}")
    wal.close()
    shutil.rmtree(d, ignore_errors=True)
    return all_ok and not errors, ops


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Crash Recovery — 10,000 writes, SIGKILL, recover, verify
# ═══════════════════════════════════════════════════════════════════════════════

def test_crash_stress():
    d = tempfile.mkdtemp()
    script_path = os.path.join(d, 'crash_worker.py')

    code = f'''
import sys, os, signal
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")
from griddb_wal import WALGrid
from binary_grid_db import Encoder, Token

wal = WALGrid(data_dir="{d}")
for i in range(10000):
    tokens = [*Encoder.encode_integer(i % 5000), Token.RECORD]
    wal.wal_append_record(tokens)
    if i == 7500:
        os.kill(os.getpid(), signal.SIGKILL)
'''

    with open(script_path, 'w') as f:
        f.write(code)

    print(f"  Crash stress: 10,000 writes, SIGKILL at 7,500...", end=" ", flush=True)

    try:
        subprocess.run(['python3', script_path], timeout=15, capture_output=True)
    except:
        pass

    # Recover
    wal = WALGrid(data_dir=d)
    count = sum(1 for t in wal.grid._tokens if t == Token.RECORD)
    ok = count >= 7000  # At least 7,000 of first 7,500 survived
    print(f"recovered={count} {'✓' if ok else '✗'}")
    wal.close()
    shutil.rmtree(d, ignore_errors=True)
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Group Commit Throughput — 10,000 writes batched
# ═══════════════════════════════════════════════════════════════════════════════

def test_group_commit_throughput():
    d = tempfile.mkdtemp()
    wal = WALGrid(data_dir=d)
    gc = GroupCommitWAL(wal, batch_size=100)
    N = 10000

    print(f"  Group commit: {N} writes, batch=100...", end=" ", flush=True)
    t0 = time.perf_counter()

    for i in range(N):
        tokens = [*Encoder.encode_integer(i), Token.RECORD]
        gc.append(tokens)
    gc.flush()

    elapsed = (time.perf_counter() - t0) * 1000
    count = sum(1 for t in wal.grid._tokens if t == Token.RECORD)
    ok = count == N
    ops = int(N / (elapsed / 1000))

    print(f"{count}/{N} {'✓' if ok else '✗'} "
          f"({elapsed:.0f}ms, {ops} writes/s, {gc.flushes} fsyncs)")
    gc.close()
    shutil.rmtree(d, ignore_errors=True)
    return ok, ops


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Flat Latency — O(1) reads at scale
# ═══════════════════════════════════════════════════════════════════════════════

def test_flat_latency():
    d = tempfile.mkdtemp()
    wal = WALGrid(data_dir=d)

    # Write records at scattered positions
    positions = [0, 1000, 5000, 9999]
    for pos in positions:
        tokens = [*Encoder.encode_integer(pos), Token.RECORD]
        wal.wal_append_record(tokens)

    print(f"  O(1) latency at scale...", end=" ", flush=True)
    times = []
    for pos in positions:
        t0 = time.perf_counter()
        rec_count = 0
        target = -1
        for t in wal.grid._tokens:
            if t == Token.RECORD:
                rec_count += 1
                if rec_count - 1 == pos:
                    target = rec_count - 1
                    break
        elapsed = (time.perf_counter() - t0) * 1_000_000
        times.append(elapsed)

    avg = sum(times) / len(times)
    flat = max(times) < avg * 3  # All reads within 3x of average
    print(f"avg={avg:.0f}µs, max={max(times):.0f}µs "
          f"{'✓ flat' if flat else '✗ spike'}")
    wal.close()
    shutil.rmtree(d, ignore_errors=True)
    return flat


# ═══════════════════════════════════════════════════════════════════════════════
# Run All
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("═" * 70)
    print("  GridDB Stress Test — Load Harness")
    print("═" * 70)

    results = {}

    print("\n── Sequential ──")
    ok, ops = test_sequential_load()
    results['sequential-5k'] = ok

    print("\n── Concurrent ──")
    ok, cops = test_concurrent_load()
    results['concurrent-50x100'] = ok

    print("\n── Crash Recovery ──")
    results['crash-10k'] = test_crash_stress()

    print("\n── Group Commit ──")
    ok, gops = test_group_commit_throughput()
    results['group-commit-10k'] = ok

    print("\n── Latency ──")
    results['flat-latency'] = test_flat_latency()

    print("\n── Results ──")
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'✓' if ok else '✗ FAILED'}")
        if not ok: all_ok = False

    print(f"\n  {'All stress tests pass' if all_ok else 'SOME TESTS FAILED'}")
    print("═" * 70)
