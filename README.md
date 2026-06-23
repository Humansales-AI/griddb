# GridDB — The Binary Grid Database

**A Unified 5‑Bit Integer Fabric with Full ACID Support**

*Version 2.5 — Position-Addressed Storage, WAL, Replication, Transactions, Indexes, Change Streams*

---

## What is this?

A database architecture built entirely upon **5‑bit binary tokens** — 32 deterministic codes that represent signed integers, text, operators, and controls. No SQL parser. No variable-length encoding. No schema. Just fixed-width tokens at absolute addresses.

Storage is a **bit‑addressable binary grid** with full ACID guarantees.

---

## Project Status

```
✅ Atomicity     — Multi-write transactions via WAL + RECORD
✅ Consistency   — Application-enforced (schema-free by design)
✅ Isolation     — `flock` per-write + `write_if` CAS cross-process
✅ Durability    — WAL + fsync + SHA-256 chain, crash recovery verified
✅ Point reads   — O(1) at absolute bit offsets
✅ Indexes       — Hash (O(1) equality) + B-tree (O(log n) range)
✅ Replication   — Master/Replica over HTTP, WAL as oplog
✅ Transactions  — Begin/Commit/Rollback, WAL-backed
✅ Change streams — SSE + long-poll from WAL tail
```

---

## Why believe this

Two provable results you can reproduce in one command:

**Cross-language determinism:** 53 test vectors — integers, bigints, mixed-case words, SPECIAL2 punctuation — produce bit-identical packed bytes in Python and TypeScript. A record written on one engine reads byte-correct on the other.

```bash
./conformance.sh   # 53 pass, 0 fail — Python ≡ TypeScript, bit for bit
```

**Native geometry on the fabric:** A transformer with a 32-token embedding table (1,024 floats) learns Manhattan distance directly from raw 5-bit token streams — no schema, no query planner, no feature engineering.

```bash
python3 examples/grid_transformer.py   # trained in seconds, queries the grid via attention
```

---

## The 32‑Token Lexicon

Three contexts, same 32 binary codes:

| Binary  | NUM | WORD | SPECIAL |
|:-------:|:---:|:----:|:-------:|
| `00000` | `0` | `A`  | `a`     |
| ... | ... | ... | ... |
| `11001` | `-9` | `Z` | `z` |
| `11010` | `^` | `␣` | `@` |
| `11011` | `S` | `.` | `-` |
| `11100` | **RECORD** | **RECORD** | **RECORD** |
| `11101` | **CHECKSUM** | **CHECKSUM** | **CHECKSUM** |
| `11110` | **END** | **END** | **END** |
| `11111` | **START** | **START→** | **START→** |

`START` in NUM → WORD. `START` in WORD → SPECIAL. Digits via context switching.

---

## ACID — How It Works

### Atomicity (Multi-Write)

```python
txn = grid.begin()
txn.put(0, alice_tokens)   # writes to WAL as PENDING
txn.put(1, bob_tokens)     # writes to WAL as PENDING
txn.commit()                # writes TXN_COMMIT → both visible
```

Writes go to WAL immediately (durable, no memory limit). TXN_COMMIT makes them visible. Crash before COMMIT → pending writes discarded on recovery.

### Consistency

Schema-free by design. The grid stores tokens — the application enforces rules. Zero metadata overhead, maximum flexibility.

### Isolation

Two complementary primitives, both cross-process:

- **`flock` per-write** — prevents data corruption. Every write acquires an exclusive file lock, fsyncs, releases. This guards the storage layer against torn writes and interleaved appends.
- **`write_if` (compare-and-swap)** — prevents double-spend. A write only commits if the record hasn't changed since you read it. If another process wrote first, CAS fails and you retry. This is the engine-level fix for lost-update bugs. Verified: 12/12 trials, zero double-spends.

The threaded sum-N test passes (zero lost updates under contention). Cross-process CAS is verified with 6 concurrent processes withdrawing from one account — exactly 1 succeeds per trial.

### Durability

Every write: WAL → `fsync()` → SHA-256 chain → eventual checkpoint. Crash recovery replays WAL, discarding uncommitted transactions.

---

## Architecture Layers

```
┌──────────────────────────────────────────────┐
│ Application Layer                            │
│  Change Streams  │  Replication  │  Queries  │
├──────────────────────────────────────────────┤
│ Index Layer                                  │
│  HashIndex (O(1))  │  BTreeIndex (O(log n)) │
├──────────────────────────────────────────────┤
│ Transaction Layer                            │
│  Begin/Commit/Rollback  │  WAL durability    │
├──────────────────────────────────────────────┤
│ Storage Layer                                │
│  AllocGrid (O(1) point)  │  PositionedGrid   │
│  BinaryGrid (append)     │  WAL+SHA256       │
└──────────────────────────────────────────────┘
```

---

## Performance

| Operation | GridDB (in-memory) | GridDB (durable) | SQLite | PostgreSQL |
|---|---|---|---|---|
| Point read (O(1) AllocGrid) | ~120µs | ~120µs | ~200µs | ~200µs |
| Write (single, fsync'd) | — | ~630µs | ~300µs | ~300µs |
| Write (group commit, batched) | — | ~630µs amortized | ~200µs amortized | ~200µs amortized |
| Range scan (1K, B-tree) | ~2ms | ~2ms | ~3ms | ~2ms |
| Hash lookup | ~150µs | ~150µs | ~200µs | ~200µs |
| Schema overhead | **0 bytes** | **0 bytes** | ~4B/row | ~4B/row |
| Deterministic encoding | ✓ | ✓ | ✗ | ✗ |
| Content-addressable | SHA-256 | SHA-256 | ✗ | ✗ |
| Geometry queries | Native | Native | ✗ | ✗ |

*Durable numbers from group commit correctness suite (~1,580 writes/s, batched fsync).*
*In-memory numbers from AllocGrid point reads (no fsync on read path).*

---

## Gap Assessment

| Feature | GridDB | MongoDB | PostgreSQL |
|---|---|---|---|
| O(1) point reads | ✓ | ✓ | ✓ |
| Secondary indexes | ✓ | ✓ | ✓ |
| Range queries | ✓ | ✓ | ✓ |
| ACID transactions | ✓ | ✓ | ✓ |
| Replication | ✓ | ✓ | ✓ |
| Change streams | ✓ | ✓ | ~ (logical dec) |
| Aggregation pipeline | — | ✓ | ✓ |
| Deterministic bytes | ✓ | ✗ | ✗ |
| Content addressing | ✓ | ✗ | ✗ |
| Zero schema overhead | ✓ | ~ | ✗ |

**What MongoDB/PostgreSQL have that GridDB doesn't:**
- Aggregation pipeline (deferred — not needed yet)
- Decades of production hardening (tooling, drivers, cloud)
- Full-text search, geospatial indexes, JSONB, window functions

**What GridDB has that they don't:**
- Bit-level determinism — same input = same bytes everywhere
- SHA-256 content addressing — verify any segment without schema
- 32-token vocabulary — 99.9% smaller embedding table (1,024 floats vs 131M for GPT-2). Tradeoff: longer sequences (attention is O(seq²)). A hash is ~64 tokens, a word is ~1 token/char. Tiny vocabulary, explicit sequences — honest about the cost.
- Geometry-native queries — no extensions needed
- Append-only audit trail — every write is a permanent record

---

## Correctness Suite

The strongest evidence GridDB works: `griddb_correctness.py` (Python) and `griddb_correctness.ts` (TypeScript).

| Test | What it proves | Result |
|---|---|---|
| Sum-N (single-thread) | Read-modify-write is atomic | ✓ zero lost updates |
| Sum-N (threaded, `threading.Lock`) | Serialized writes correct under contention | ✓ zero lost updates |
| Crash recovery (SIGKILL) | Data survives hard kill mid-write | ✓ WAL replays on restart |
| Group commit (batched fsync) |Throughput scaling via batch durability | ✓ ~1,580 writes/s |
| WAL checkpoint | Bounded disk growth, data survives snapshot | ✓ |
| Tombstone | Soft delete + alloc flags correct | ✓ |

```bash
cd python
python3 griddb_correctness.py   # Python correctness suite
cd ../typescript
npx tsx tests/griddb_correctness.ts  # TypeScript correctness suite
```

---

## Project Structure

```
griddb/
├── README.md
├── python/
│   ├── binary_grid_db.py          # Core: tokens, encoder, parser, 3 contexts
│   ├── griddb_wal.py              # WAL + SHA-256 chaining
│   ├── griddb_positioned.py       # O(1) positioned grid
│   ├── griddb_alloc.py            # AllocGrid (billions of records)
│   ├── griddb_index.py            # HashIndex + BTreeIndex
│   ├── griddb_replication.py      # Master/Replica HTTP sync
│   ├── griddb_transactions.py     # ACID via WAL + RECORD
│   ├── griddb_changestream.py     # SSE/long-poll from WAL
│   ├── griddb_correctness.py      # Correctness suite (sum-N, crash, group commit)
│   ├── test_binary_grid_db.py     # 168 unit tests
│   └── requirements.txt
├── typescript/
│   ├── src/                       # Full TS port (15 modules)
│   └── tests/
│       └── griddb_correctness.ts  # TS correctness suite
└── examples/
    ├── griddb_explorer.py
    └── grid_transformer.py
```

## Quick Start

```bash
cd python
python3 griddb_correctness.py     # Correctness suite — prove it works
python3 -m unittest test_binary_grid_db -v  # 168 unit tests

# Individual demos
python3 griddb_alloc.py           # O(1) reads at scale
python3 griddb_index.py           # Hash + B-tree indexes
python3 griddb_replication.py     # Master/replica sync
python3 griddb_transactions.py    # ACID transactions
python3 griddb_changestream.py    # Change streams
```

---

## License

MIT

---

*"The grid stores tokens, not tables. Consumers decide meaning — expressions, tuples, words, or anything else. This is the Unix philosophy applied to data persistence."*
