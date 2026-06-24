# 5bit — The Binary Grid Database

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

Four contexts, same 32 binary codes. 28 mappable slots each (00000–11011). Four controls (11100–11111) retain meaning everywhere.

| Binary  | NUM | WORD | SPECIAL | SPECIAL2 | SPECIAL3 |
|:-------:|:---:|:----:|:-------:|:--------:|:--------:|
| `00000` | `0` | `A`  | `a`     | `!`      | `AUTH`   |
| `00001` | `1` | `B`  | `b`     | `"`      | `GRANT_R`|
| `00010` | `2` | `C`  | `c`     | `#`      | `GRANT_W`|
| `00011` | `3` | `D`  | `d`     | `$`      | `REVOKE` |
| `00100` | `4` | `E`  | `e`     | `%`      | `ENCRYPT`|
| `00101` | `5` | `F`  | `f`     | `&`      | `—`      |
| `00110` | `6` | `G`  | `g`     | `'`      | `—`      |
| `00111` | `7` | `H`  | `h`     | `(`      | `—`      |
| `01000` | `8` | `I`  | `i`     | `)`      | `—`      |
| `01001` | `9` | `J`  | `j`     | `*`      | `—`      |
| `01010` | `+` | `K`  | `k`     | `+`      | `—`      |
| `01011` | `-` | `L`  | `l`     | `,`      | `—`      |
| `01100` | `*` | `M`  | `m`     | `/`      | `—`      |
| `01101` | `/` | `N`  | `n`     | `:`      | `—`      |
| `01110` | `=` | `O`  | `o`     | `;`      | `—`      |
| `01111` | `(` | `P`  | `p`     | `<`      | `—`      |
| `10000` | `)` | `Q`  | `q`     | `=`      | `—`      |
| `10001` | `-1`| `R`  | `r`     | `>`      | `—`      |
| `10010` | `-2`| `S`  | `s`     | `?`      | `—`      |
| `10011` | `-3`| `T`  | `t`     | `[`      | `—`      |
| `10100` | `-4`| `U`  | `u`     | `\`      | `—`      |
| `10101` | `-5`| `V`  | `v`     | `]`      | `—`      |
| `10110` | `-6`| `W`  | `w`     | `^`      | `—`      |
| `10111` | `-7`| `X`  | `x`     | `_`      | `—`      |
| `11000` | `-8`| `Y`  | `y`     | `` ` ``  | `—`      |
| `11001` | `-9`| `Z`  | `z`     | `{`      | `—`      |
| `11010` | `^` | `␣`  | `@`     | `\|`     | `—`      |
| `11011` | `S` | `.`  | `-`     | `}`      | `—`      |
| `11100` | **RECORD** | **RECORD** | **RECORD** | **RECORD** | **RECORD** |
| `11101` | **CHECKSUM** | **CHECKSUM** | **CHECKSUM** | **CHECKSUM** | **CHECKSUM** |
| `11110` | **END** | **END** | **END** | **END** | **END** |
| `11111` | **START** | **START** | **START** | **START** | **START** |

### Context Switching — How It Works

The parser starts in **NUM** state. Four control tokens navigate the stack:

| Token | Current state | Action |
|:-----:|:--------------|:-------|
| `START` (11111) | NUM | Enter WORD |
| `START` (11111) | WORD | Enter SPECIAL |
| `START` (11111) | SPECIAL | Enter SPECIAL2 |
| `START` (11111) | SPECIAL2 | Enter SPECIAL3 (control commands) |
| `END` (11110) | SPECIAL3 | Pop to SPECIAL2 |
| `END` (11110) | SPECIAL2 | Pop to SPECIAL |
| `END` (11110) | SPECIAL | Pop to WORD |
| `END` (11110) | WORD | Pop to NUM (finalize) |
| `RECORD` (11100) | Any | Finalize, emit record boundary, pop to NUM |
| `CHECKSUM` (11101) | Any | Emit integrity marker |

**SPECIAL3 — Control Commands (token-level RLS):**

The 28 slots at SPECIAL3 depth map to control commands instead of characters:
`AUTH(uid)`, `GRANT_R(uid)`, `GRANT_W(uid)`, `REVOKE(uid)`, `ENCRYPT(key_id)`.

```
"AUTH user 42" → START×4  CMD_AUTH  D4 D2 END  END×5    (5 STARTs to reach SPECIAL3)
```

These are permission *representations* in the token fabric. Combine with encryption for enforcement.

**Encoding examples:**

```
"HI"    → START  H  I  END                              (2 letters, WORD)
"hi"    → START  START  h  i  END  END                   (2 lowercase, SPECIAL)
"Hi"    → START  H  START  i  END  END                   (mixed case)
"a!b"   → START  START  a  START  !  END  b  END  END    (SPECIAL2 punctuation)
"a@b"   → START  START  a  @  b  END  END                (@ stays in SPECIAL)
"a.b"   → START  a  .  b  END                            (. in WORD, no context switch)
```

Digits (0-9) encode by temporarily popping to NUM: `END END D3 START` — pop SPECIAL→WORD→NUM, emit digit, re-enter WORD. A 64-char hex hash with digits and letters costs ~70-170 tokens depending on digit density.

The **RECORD** token (11100) terminates logical tuples. Everything between two RECORD tokens is one record. This is the boundary for geometric queries — Manhattan distance compares record vectors.

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

SQLite has 50 years of optimization, a page cache, B-tree, and WAL mode. 5bit is an append-only 5-bit fabric with no page cache and per-read fd opens. Honest numbers:

| Operation | 5bit | SQLite (WAL) |
|---|---|---|
| Point read (O(1) alloc) | ~120µs uncached / ~2µs cached (LRU) | ~50µs (page cache) |
| Write (group commit, fsync'd) | ~630µs amortized (~1,580/s) | ~50µs amortized (~20,000/s) |
| Compaction | Manual, O(n) scan | Auto, background |
| Schema overhead | **0 bytes** | ~4 bytes/row |
| Deterministic encoding | ✓ (SHA-256 content-addressed) | ✗ |
| Geometry queries (Manhattan/Hamming) | Native | ✗ (requires app code) |
| Cross-language determinism | ✓ (53/53 byte-identical Python≡TS) | ✗ |
| Audit trail | Append-only, every write permanent | ✗ (VACUUM reclaims) |

5bit is not faster than SQLite. It's deterministic, content-addressed, and schema-free — properties SQLite physically cannot provide. Different tools for different jobs.

---

## Gap Assessment

| Feature | 5bit | MongoDB | PostgreSQL |
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

**What MongoDB/PostgreSQL have that 5bit doesn't:**
- Aggregation pipeline (deferred — not needed yet)
- Decades of production hardening (tooling, drivers, cloud)
- Full-text search, geospatial indexes, JSONB, window functions

**What 5bit has that they don't:**
- Bit-level determinism — same input = same bytes everywhere
- SHA-256 content addressing — verify any segment without schema
- 32-token vocabulary — 99.9% smaller embedding table (1,024 floats vs 131M for GPT-2). Tradeoff: longer sequences (attention is O(seq²)). A hash is ~64 tokens, a word is ~1 token/char. Tiny vocabulary, explicit sequences — honest about the cost.
- Geometry-native queries — no extensions needed
- Append-only audit trail — every write is a permanent record

---

## Correctness Suite

The strongest evidence 5bit works: `5bit_correctness.py` (Python) and `5bit_correctness.ts` (TypeScript).

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
python3 5bit_correctness.py   # Python correctness suite
cd ../typescript
npx tsx tests/5bit_correctness.ts  # TypeScript correctness suite
```

---

## Project Structure

```
5bit/
├── README.md
├── python/
│   ├── binary_grid_db.py          # Core: tokens, encoder, parser, 3 contexts
│   ├── 5bit_wal.py              # WAL + SHA-256 chaining
│   ├── 5bit_positioned.py       # O(1) positioned grid
│   ├── 5bit_alloc.py            # AllocGrid (billions of records)
│   ├── 5bit_index.py            # HashIndex + BTreeIndex
│   ├── 5bit_replication.py      # Master/Replica HTTP sync
│   ├── 5bit_transactions.py     # ACID via WAL + RECORD
│   ├── 5bit_changestream.py     # SSE/long-poll from WAL
│   ├── 5bit_correctness.py      # Correctness suite (sum-N, crash, group commit)
│   ├── test_binary_grid_db.py     # 168 unit tests
│   └── requirements.txt
├── typescript/
│   ├── src/                       # Full TS port (15 modules)
│   └── tests/
│       └── 5bit_correctness.ts  # TS correctness suite
└── examples/
    ├── 5bit_explorer.py
    └── grid_transformer.py
```

## Quick Start

```bash
cd python
python3 5bit_correctness.py     # Correctness suite — prove it works
python3 -m unittest test_binary_grid_db -v  # 168 unit tests

# Individual demos
python3 5bit_alloc.py           # O(1) reads at scale
python3 5bit_index.py           # Hash + B-tree indexes
python3 5bit_replication.py     # Master/replica sync
python3 5bit_transactions.py    # ACID transactions
python3 5bit_changestream.py    # Change streams
```

---

## License

MIT

---

*"The grid stores tokens, not tables. Consumers decide meaning — expressions, tuples, words, or anything else. This is the Unix philosophy applied to data persistence."*
