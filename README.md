# 5bit — The 5‑Bit Deterministic Binary Database

**Same bytes everywhere. Python ≡ TypeScript. 53/53 cross-language conformance, zero mismatches.**

```
npm install fivebit-client
```

```typescript
import { createClient } from 'fivebit-client'

const db = createClient({ url: 'http://localhost:8080' })

// Auth — email/password + OAuth (Google, GitHub)
await db.auth.signUp('alice@example.com', 'password123')
await db.auth.signInWithOAuth('google', { idToken: googleToken })

// Typed CRUD with ETag caching + conditional GETs
const users = db.from('users')
await users.insert({ name: 'Alice', balance: 100 })
const alice = await users.getById(1)  // 304 on repeat

// Aggregates — count, sum, avg, min, max with filters
await users.count('age:gt:21')
await users.avg('balance', 'age:gt:21')

// Content-addressed storage — SHA-256 dedup
const bucket = db.storage.bucket('avatars')
await bucket.upload('alice.png', fileBuffer)

// Realtime — WebSocket with presence + channels
db.onChanges('users', (record) => console.log('changed:', record))
const chat = db.channel('room')
chat.broadcast({ text: 'hello' })
chat.presence(1, 'Alice')  // "who's online"
```

---

## What is this?

A database architecture built entirely upon **5‑bit binary tokens**. 32 deterministic codes that represent signed integers, English text, arithmetic operators, decimal scaling, and control commands — all within a single flat, append-only bitstream. No SQL parser. No variable-length encoding. No floating-point mantissas. No schema.

Storage is a **bit‑addressable binary grid**. Records live at known positions: `record_id × STRIDE` bits. Access is O(1) — seek + read, no B-tree, no index, no scan. Relationships between records are derived natively via **Hamming distance** (address proximity) and **Manhattan distance** (value proximity), rendering traditional relational joins and indexes obsolete.

---

## Why believe this

Two provable results you can reproduce in one command:

**Cross-language determinism:** 53 test vectors — integers, bigints, mixed-case words, SPECIAL2 punctuation, SPECIAL3 commands — produce bit-identical packed bytes in Python and TypeScript. A record written on one engine reads byte-correct on the other. Same input = same bytes everywhere.

```bash
./conformance.sh   # 53 pass, 0 fail — Python ≡ TypeScript, bit for bit
```

**Canonical compaction:** Write the same history on both engines, compact them, and the resulting files are byte-identical (SHA-256 `090def36…` on both sides). Same logical content → same bytes, any engine, any language.

```bash
./verify.sh   # conformance + canonical compaction, exits non-zero on mismatch
```

**Native geometry on the fabric:** A transformer with a 32-token embedding table (1,024 floats) learns Manhattan distance directly from raw 5-bit token streams — no schema, no query planner, no feature engineering.

```bash
python3 examples/grid_transformer.py   # trained in seconds, queries the grid via attention
```

---

## Project Status

```
✅ Atomicity        — Multi-write transactions via WAL + RECORD
✅ Consistency      — Application-enforced (schema-free by design)
✅ Isolation        — flock per-write + write_if CAS cross-process
✅ Durability       — WAL + fsync + SHA-256 chain, crash recovery verified
✅ Point reads      — O(1) at absolute bit offsets (AllocGrid)
✅ Indexes          — Hash (O(1) equality) + B-tree (O(log n) range)
✅ Replication      — Master/Replica over HTTP, WAL as oplog
✅ Transactions     — Begin/Commit/Rollback, WAL-backed, lock-spanned
✅ Change streams   — SSE + long-poll from WAL tail
✅ REST API         — Deterministic routes, content-addressed ETags
✅ Auth             — PBKDF2 + JWT sessions + OAuth (Google/GitHub)
✅ Storage          — Content-addressed file store, SHA-256 dedup
✅ Realtime         — WebSocket + presence + broadcast channels
✅ npm client       — fivebit-client@0.2.1
✅ Conformance      — 53/53 cross-language + canonical compaction
✅ Compaction       — Reclaim tombstone space, crash-atomic
✅ Page cache       — LRU read cache, 68× hit speedup (wired into AllocGrid)
✅ Multi-mode auth  — Zero-knowledge (server blind) or Managed (server can read)
```

---

## The 32‑Token Lexicon — Five Contexts

Four contexts, same 32 binary codes. 28 mappable slots each (00000–11011). Four controls (11100–11111) retain meaning across all contexts. SPECIAL3 maps the same 28 slots to control commands instead of characters — token-level permissions in the fabric itself.

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

---

## Context Switching — How It Works

The parser starts in **NUM** state. Four control tokens navigate a 4-level stack:

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

**Encoding examples:**

```
"HI"    → START  H  I  END                                     (2 letters, WORD)
"hi"    → START  START  h  i  END  END                          (2 lowercase, SPECIAL)
"Hi"    → START  H  START  i  END  END                          (mixed case)
"a!b"   → START  START  a  START  !  END  b  END  END           (SPECIAL2 punctuation)
"a@b"   → START  START  a  @  b  END  END                       (@ stays in SPECIAL)
"a.b"   → START  a  .  b  END                                   (. in WORD, no switch)
"AUTH 42" → START×4  CMD_AUTH  D4 D2 END  END×5                (SPECIAL3 command)
```

Digits (0-9) encode by temporarily popping to NUM: `END END D3 START` — pop SPECIAL→WORD→NUM, emit digit, re-enter WORD. A 64-char hex hash with digits and letters costs ~70-170 tokens depending on digit density.

The **RECORD** token (11100) terminates logical tuples. Everything between two RECORD tokens is one record. This is the boundary for geometric queries — Manhattan distance compares record vectors.

**SPECIAL3 Control Commands** map 28 slots to token-level access control primitives:

```
AUTH(user_id)     — declare record owner at token level
GRANT_R(user_id)  — grant read access to user
GRANT_W(user_id)  — grant write access to user
REVOKE(user_id)   — revoke access from user
ENCRYPT(key_id)   — mark record as encrypted with key
```

21 reserved slots remain for future commands. CMD_LABEL (00101) tags cell positions with metadata — the data describes itself:

```
LABEL 0 "user_id"    START×4 D5 D0 END×4  START u s e r _ i d END END
LABEL 1 "age"        START×4 D5 D1 END×4  START a g e END END
LABEL 2 "email"      START×4 D5 D2 END×4  START e m a i l END END
LABEL 3 "balance"    START×4 D5 D3 END×4  START b a l a n c e END END
LABEL 4 "name"       START×4 D5 D4 END×4  START n a m e END END
LABEL 5 "created"    START×4 D5 D5 END×4  START c r e a t e d END END

DATA:  D1 END  D2 D5 END  a l i c e @ d e m o . c o m END END  D5 D0 D0 D0 END  A l i c e END END  D8 END  RECORD
       uid=1   age=25       email="alice@demo.com"                  balance=5000          name="Alice"      created=8
```

B-tree reads labels → finds "age" at position 1 → indexes every record's position-1 value. No external schema config.

---

## Label-Driven Architecture

Labels replace schemas. The data describes itself.

**Without labels** (spec-driven):
```python
# Server needs external config
spec = {'fields': ['age', 'balance', 'name']}
# B-tree indexes position 0 because spec says "age is at 0"
```

**With labels** (data-driven):
```python
# Labels baked into the token stream
Encoder.encode_label(0, "age")       # cell 0 tagged "age"
Encoder.encode_label(1, "balance")   # cell 1 tagged "balance"
Encoder.encode_label(2, "name")      # cell 2 tagged "name"

# B-tree auto-discovers: position 0 = age, position 1 = balance
# Query ?filter=age:gt:21 — found via label lookup, no config
```

The server scans for CMD_LABEL tokens on startup, builds a field_name → position map, and auto-creates B-tree indexes. Labels travel with the data. Drop a labeled grid file on any server and it knows the schema instantly.

---

## Arithmetic — Signed Digits + Shunting-Yard

All numbers are encoded as **signed-digit tokens**. Each digit carries its own sign. No floating-point. No IEEE 754.

**Integers**: `123` → `D1 D2 D3 END`. `-123` → `N1 N2 N3 END`. `0` → `D0 END`.

**Decimal scaling**: store as integer with an `S` (Scale) annotation. `12.50` → `D1 D2 D5 D0 END T_SCALE N2 END` = "1250 with 2 decimal places." Division-free — all arithmetic is integer. Rounding is explicit.

**Shunting-Yard expression parser**: `3 + 4 * 2` → `D3 END D4 END D2 END * +` (postfix). The parser evaluates in O(n) with a stack. Operators: `+ - * / ( ) = ^`. The `^` is token T_POW (11010 in NUM context).

```
Expression:  (1 + 2) * 3
Tokens:      T_LPAREN D1 END T_PLUS D2 END T_RPAREN T_MUL D3 END
Postfix:     1 2 + 3 *
Result:      9
```

**Geometric context**: Hamming distance compares raw token bits. Manhattan distance sums absolute differences of value vectors across records. Both are O(n) on the token stream. No index required.

---

## Delimiters — RECORD, END, START

Three structural tokens define the fabric:

| Token | Binary | Purpose |
|:-----:|:------:|:--------|
| `RECORD` | 11100 | Terminates a logical tuple. Everything between two RECORDs is one record. The boundary for geometric queries. |
| `END` | 11110 | Terminates a number or word. Also pops the context stack (SPECIAL3→SPECIAL2→SPECIAL→WORD→NUM). |
| `START` | 11111 | Pushes the context stack (NUM→WORD→SPECIAL→SPECIAL2→SPECIAL3). |

**Field separation**: Numbers self-terminate with END. Words self-terminate with END. So fields are naturally separated:

```
NUM(25) END  NUM(1000) END  START A l i c e END END  RECORD
   ↑              ↑                    ↑               ↑
  age=25      balance=1000         name="Alice"    end of record
```

No comma, no tab, no JSON delimiter. The token stream IS the format. A parser that understands END and RECORD can parse any 5bit data without a schema.

---

## Architecture Layers

```
┌──────────────────────────────────────────────┐
│ Application Layer                            │
│  REST API  │  Auth  │  Storage  │  Realtime  │
├──────────────────────────────────────────────┤
│ Fivebit Libraries (optional, zero core changes) │
│  RLS Engine  │  CryptoRLS  │  PerUserGrid    │
│  CommandRLS  │  MultiMode  │  TenantGrid     │
├──────────────────────────────────────────────┤
│ Index Layer                                  │
│  HashIndex (O(1))  │  BTreeIndex (O(log n)) │
├──────────────────────────────────────────────┤
│ Transaction Layer                            │
│  Begin/Commit/Rollback  │  WAL durability    │
├──────────────────────────────────────────────┤
│ Storage Layer                                │
│  AllocGrid (O(1) point)  │  PositionedGrid   │
│  LRU Page Cache          │  WAL + SHA-256    │
└──────────────────────────────────────────────┘
```

---

## ACID — How It Works

### Atomicity (Multi-Write)

```python
txn = grid.begin()
txn.put(0, alice_tokens)   # writes to WAL as PENDING
txn.put(1, bob_tokens)     # writes to WAL as PENDING
txn.commit()                # writes TXN_COMMIT → both visible
```

Lock spans entire transaction: `begin()` acquires flock → reads happen → `commit()` writes → lock releases. No process can interleave. Crash recovery: DIRTY marker ensures torn transactions are re-applied by the next survivor.

### Consistency

Schema-free by design. The grid stores tokens — the application enforces rules. Zero metadata overhead, maximum flexibility.

### Isolation

Two complementary primitives, both cross-process:

- **`flock` per-write** — prevents data corruption. Every write acquires an exclusive file lock, fsyncs, releases.
- **`write_if` (compare-and-swap)** — prevents double-spend. A write only commits if the record hasn't changed since you read it. Reentrant-lock with depth counter ensures nested CAS doesn't release prematurely.

Verified: 24 real OS processes, 1200 increments on one hot account, zero lost updates (`griddb_concurrency_cas.py` exits 0). Transaction lock spans full read→commit, serialized cross-process.

### Durability

Every write: WAL → `fsync()` → SHA-256 chain → eventual checkpoint. Crash recovery replays WAL, discards uncommitted transactions. DIRTY marker ensures committed-but-unapplied transactions are finished by survivors on `begin()`.

---

## Performance (Python)

| Operation | 5bit |
|---|---|
| Point read (cached, warm) | ~3µs |
| Point read (uncached) | ~44µs |
| Point read (thrash, exceeds cache) | ~95µs |
| Write (group commit, batched fsync) | ~48µs (~20,800/s) |
| Compaction | Manual O(n), crash-atomic |
| Deterministic encoding | ✓ (SHA-256 content-addressed) |
| Geometry queries | Native (Manhattan, Hamming) |
| Cross-language determinism | ✓ (53/53 Python≡TS) |
| Audit trail | Append-only, every write permanent |

*Cached reads hit ~3µs when the working set fits in the LRU cache. Enable with `AllocGrid("./data", cache_size=1000)`.*

---

## Correctness Suite

The strongest evidence 5bit works:

| Test | Proves | Result |
|---|---|---|
| Sum-N single-thread | RMW atomic | ✓ zero lost |
| Sum-N threaded | Serialized correct | ✓ zero lost |
| Crash recovery (SIGKILL) | Data survives hard kill | ✓ WAL replays |
| Group commit (batched fsync) | Throughput scaling | ✓ ~20,800/s |
| WAL checkpoint | Bounded disk | ✓ |
| Multi-process CAS (24 procs) | Cross-process atomic | ✓ zero lost |
| Canonical compaction | Deterministic lifecycle | ✓ Python≡TS |

```bash
python3 griddb_correctness.py     # Python
npx tsx tests/griddb_correctness.ts  # TypeScript
python3 griddb_concurrency_cas.py    # Multi-process CAS
./verify.sh                          # Conformance + compaction
```

---

## Project Structure

```
5bit/
├── python/                          Core engine
│   ├── binary_grid_db.py            Tokens, encoder, parser, 5 contexts
│   ├── griddb_alloc.py              AllocGrid (O(1) reads, LRU cache, compaction)
│   ├── griddb_wal.py                WAL + SHA-256 chaining
│   ├── griddb_positioned.py         PositionedGrid (O(1) by stride)
│   ├── griddb_index.py              HashIndex + BTreeIndex
│   ├── griddb_transactions.py       ACID transactions (lock-spanned, DIRTY recovery)
│   ├── griddb_replication.py        Master/Replica HTTP
│   ├── griddb_changestream.py       SSE + long-poll from WAL
│   ├── griddb_correctness.py        Correctness suite
│   ├── griddb_stress.py             Stress test harness
│   ├── griddb_concurrency_cas.py    Multi-process CAS regression
│   └── test_binary_grid_db.py       168 unit tests
├── typescript/
│   ├── src/                         Full TS port
│   │   ├── types.ts                 32 Token enum, ParserState, ParsedToken types
│   │   ├── tokens.ts                5-bit mappings (NUM/WORD/SPECIAL/SPECIAL2 + control)
│   │   ├── encoder.ts               Signed-digit integers, words, expressions, records
│   │   ├── parser.ts                FSM parser (NUM→WORD→SPECIAL→SPECIAL2→SPECIAL3)
│   │   ├── serialization.ts         5-bit ↔ 8-bit pack/unpack
│   │   ├── arithmetic.ts            Shunting-Yard + decimal arithmetic
│   │   ├── geometry.ts              Hamming/Manhattan distance on token streams
│   │   ├── checksum.ts              Modulo-32 integrity checks
│   │   ├── grid.ts                  BinaryGrid (append-only)
│   │   ├── alloc.ts                 AllocGrid (O(1) reads, LRU cache, compaction, WAL, groups)
│   │   ├── positioned.ts            PositionedGrid (O(1) by stride)
│   │   ├── indexes.ts               HashIndex + BTreeIndex
│   │   ├── replication.ts           Master/Replica (cross-process)
│   │   ├── transactions.ts          ACID transactions (lock-spanned, DIRTY recovery)
│   │   ├── changestream.ts          SSE + long-poll event stream
│   │   ├── server.ts                Standalone REST API server (no Python needed)
│   │   └── fivebit/                 Optional libs (auth, RLS, crypto, per_user, tenant, cache, commands)
│   ├── client/                      npm package (fivebit-client@0.2.2)
│   └── tests/                       48 Jest + 5 correctness + conformance
├── fivebit/                         Optional libraries (zero core changes)
│   ├── auth/                        PBKDF2 + sessions + MultiMode (zero/managed)
│   ├── rls/                         RLSEngine, CryptoRLS, PerUserGrid, CommandRLS
│   ├── tenant/                      TenantGrid (stable SHA-256 hash)
│   ├── cache.py                     LRU page cache (wired into AllocGrid)
│   └── api/                         REST API, Auth server, OAuth, Storage, Realtime
├── conformance.sh                   53/53 cross-language determinism
├── verify.sh                        Conformance + canonical compaction
└── examples/                        Transformer, explorer, benchmarks
```

---

## Quick Start

```bash
git clone https://github.com/Humansales-AI/5bit && cd 5bit

# Verify
./verify.sh

# Start API server
python3 -c "
from fivebit.api.server import APIServer
APIServer('./data', {'name':'users','fields':['balance','name']}, port=8080).start(True)
"

# Client
npm install fivebit-client
```

---

## License

MIT
