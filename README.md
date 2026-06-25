# 5bit — The 5‑Bit Deterministic Database

**Same bytes everywhere. Python ≡ TypeScript. 53/53 cross-language conformance, zero mismatches.**

```
npm install fivebit-client
```

```typescript
import { createClient } from 'fivebit-client'
const db = createClient({ url: 'http://localhost:8080' })
await db.auth.signUp('alice@example.com', 'password123')
await db.auth.signIn('alice@example.com', 'password123')
const users = db.from('users')
await users.insert({ name: 'Alice', balance: 100 })
```

---

## Quick Start

```bash
# Clone + verify
git clone https://github.com/Humansales-AI/5bit
cd 5bit
./verify.sh                    # 53/53 cross-language determinism + canonical compaction

# Start API server
python3 -c "
from fivebit.api.server import APIServer
s = APIServer('./data', {'name':'users','fields':['balance','name']}, port=8080)
s.start(blocking=True)
"

# Client
npm install fivebit-client
```

---

## What is this?

A database built on 32 five-bit tokens. No SQL. No schema. No floats.
Four contexts × 28 slots = 112 encodable symbols. Everything is deterministic.

| Property | 5bit | SQLite | PostgreSQL | Supabase |
|---|---|---|---|---|
| Deterministic bytes | ✓ SHA-256 | ✗ | ✗ | ✗ |
| Cross-language (Python≡TS) | ✓ 53/53 | ✗ | ✗ | ✗ |
| Content-addressed | ✓ | ✗ | ✗ | ✗ |
| Schema-free | ✓ | ✗ | ✗ | ✗ |
| npm install | ✓ | — | — | ✓ |
| Realtime | ✓ SSE | ✗ | ✗ | ✓ |
| O(1) point reads | ✓ | ✓ | ✓ | ✓ |
| B-tree indexes | ✓ | ✓ | ✓ | ✓ |
| WAL + crash recovery | ✓ | ✓ | ✓ | ✓ |

---

## Performance (honest)

| Operation | 5bit | SQLite |
|---|---|---|
| Point read (cached, warm) | ~3µs | ~6µs |
| Point read (uncached) | ~44µs | ~6µs |
| Write (group commit) | ~20,800/s | ~20,000/s |
| Compaction | Manual O(n) | Auto background |

---

## The 5‑Context Lexicon

| Binary | NUM | WORD | SPECIAL | SPECIAL2 | SPECIAL3 |
|:------:|:---:|:----:|:-------:|:--------:|:--------:|
| 00000 | 0 | A | a | ! | AUTH |
| 00001 | 1 | B | b | " | GRANT_R |
| 00010 | 2 | C | c | # | GRANT_W |
| 00011 | 3 | D | d | $ | REVOKE |
| 00100 | 4 | E | e | % | ENCRYPT |
| ... | ... | ... | ... | ... | — |
| 11011 | S | . | - | } | — |
| 11100 | RECORD | RECORD | RECORD | RECORD | RECORD |
| 11101 | CHECKSUM | CHECKSUM | CHECKSUM | CHECKSUM | CHECKSUM |
| 11110 | END | END | END | END | END |
| 11111 | START | START | START | START | START |

Context switching: `START` pushes deeper. `END` pops back. `RECORD` terminates.

---

## Project

```
5bit/
├── python/          Core engine (encoder, parser, AllocGrid, WAL, indexes, replication)
├── typescript/      Full TS port (15 modules, 48 Jest + 5 correctness tests)
├── fivebit/         Optional libraries (auth, RLS, crypto, multi-tenant, API server)
├── conformance.sh   53/53 cross-language determinism
├── verify.sh        Conformance + canonical compaction
└── examples/        Transformer demo, explorer, benchmarks
```

---

## License

MIT
