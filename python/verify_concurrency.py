#!/usr/bin/env python3
"""
Lock-free concurrent-read proof (Python engine).

Two separate OS processes hit the SAME grid:
  - a writer holding flock(LOCK_EX), hammering writes
  - a reader taking NO lock at all

Each record stores [v, v] — two numbers that must always be equal. Because
data is appended at a NEW offset and fsync'd BEFORE the 16-byte alloc entry
is flipped to point at it (and old data is never overwritten), a concurrent
reader must see EITHER old-pointer->old-data OR new-pointer->new-data — never
a torn mix. We assert: zero pair-mismatches, zero parse errors.

Exit 0 on success, 1 on any torn/inconsistent read.
"""
import os
import sys
import time
import tempfile
import shutil
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from griddb_alloc import AllocGrid          # noqa: E402
from binary_grid_db import Encoder, Token   # noqa: E402

DUR = 3.0   # seconds
K = 50      # number of record ids hammered


def _writer(data_dir, stop):
    g = AllocGrid(data_dir=data_dir, cache_size=0)
    v = 1
    while not stop.value:
        g.write(v % K, Encoder.encode_record(v, v))  # invariant: [v, v]
        v += 1
    g.close()


def _reader(data_dir, stop, result):
    g = AllocGrid(data_dir=data_dir, cache_size=0)
    reads = torn = parse_err = 0
    time.sleep(0.2)  # let the writer get going
    while not stop.value:
        rec = None
        try:
            rec = g.read(reads % K)
        except Exception:
            parse_err += 1
            reads += 1
            continue
        reads += 1
        if rec is None:
            continue
        # ParsedNumber has .value; exclude control Tokens (IntEnum also has .value)
        nums = [p.value for p in rec.parsed
                if getattr(p, "value", None) is not None and not isinstance(p, Token)]
        if len(nums) != 2:
            parse_err += 1
        elif nums[0] != nums[1]:
            torn += 1
    g.close()
    result["reads"], result["torn"], result["parse_err"] = reads, torn, parse_err


def main() -> int:
    data_dir = tempfile.mkdtemp(prefix="verify_concur_")
    try:
        AllocGrid(data_dir=data_dir, cache_size=0).write(0, Encoder.encode_record(0, 0))
        mgr = mp.Manager()
        stop = mgr.Value("b", False)
        result = mgr.dict()
        w = mp.Process(target=_writer, args=(data_dir, stop))
        r = mp.Process(target=_reader, args=(data_dir, stop, result))
        w.start(); r.start()
        time.sleep(DUR)
        stop.value = True
        w.join(); r.join()

        reads = result.get("reads", 0)
        torn = result.get("torn", 1)
        parse_err = result.get("parse_err", 1)
        ok = (torn == 0 and parse_err == 0 and reads > 0)
        print("  [py] %d lock-free reads during concurrent writes" % reads)
        print("  [py] torn (pair mismatch): %d   parse errors: %d" % (torn, parse_err))
        print("  [py] lock-free snapshot reads : %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
