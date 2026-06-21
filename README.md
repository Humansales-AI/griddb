# GridDB — The Binary Grid Database

**A Unified 5‑Bit Integer Fabric for Numbers, Words, Arithmetic, and Geometric Queries**

*Version 2.0 — Foundational Specification with Record Boundaries & Integrity*

---

## What is this?

A database architecture built entirely upon **5‑bit binary tokens**. No variable‑length encodings. No floating‑point mantissas. No SQL parser. Just 32 deterministic codes that represent signed integers, English text, arithmetic operators, and decimal scaling — all within a single flat, append‑only bitstream.

Storage is a **bit‑addressable binary grid**. Relationships between records are derived natively via **Hamming distance** (address proximity) and **Manhattan distance** (value proximity), rendering traditional relational joins and indexes obsolete.

## Why?

Modern databases are built on layers of legacy abstraction: byte‑addressable storage, variable‑length UTF‑8 strings, IEEE‑754 floats, and SQL parsers. We strip all of that away to return to the fundamental essence of computation: **fixed‑width binary integers stored at absolute addresses**.

This architecture is uniquely suited for:
- Embedded systems & edge AI (deterministic, no parser overhead)
- Append‑only event logs & write‑ahead journals
- Content‑addressed storage (SHA‑256 over grid segments)
- High‑performance geometric queries (nearest‑neighbor, similarity search)
- Machine‑native data for transformer models (32‑token vocabulary = 2,048 embedding floats)

## The 32‑Token Lexicon

| Binary  | NUM context | WORD context |
|:-------:|:-----------:|:------------:|
| `00000` | `0`         | `A`          |
| `00001` | `1`         | `B`          |
| `00010` | `2`         | `C`          |
| `00011` | `3`         | `D`          |
| `00100` | `4`         | `E`          |
| `00101` | `5`         | `F`          |
| `00110` | `6`         | `G`          |
| `00111` | `7`         | `H`          |
| `01000` | `8`         | `I`          |
| `01001` | `9`         | `J`          |
| `01010` | `+`         | `K`          |
| `01011` | `-`         | `L`          |
| `01100` | `*`         | `M`          |
| `01101` | `/`         | `N`          |
| `01110` | `=`         | `O`          |
| `01111` | `(`         | `P`          |
| `10000` | `)`         | `Q`          |
| `10001` | `-1`        | `R`          |
| `10010` | `-2`        | `S`          |
| `10011` | `-3`        | `T`          |
| `10100` | `-4`        | `U`          |
| `10101` | `-5`        | `V`          |
| `10110` | `-6`        | `W`          |
| `10111` | `-7`        | `X`          |
| `11000` | `-8`        | `Y`          |
| `11001` | `-9`        | `Z`          |
| `11010` | `^`         | ` ` (Space)  |
| `11011` | `S` (Scale) | `.`          |
| `11100` | **RECORD**  | **RECORD**   |
| `11101` | **CHECKSUM**| **CHECKSUM** |
| `11110` | **END**     | **END**      |
| `11111` | **START**   | **START**    |

## Quick Examples

### Numbers (signed‑digit encoding)
```
 123 → 00001 00010 00011 11110
-123 → 10001 10010 10011 11110
   0 → 00000 11110
```

### Words (START/END delimited)
```
"HI" → 11111 00111 01000 11110
```

### Records (RECORD terminated)
```
(1, 2) → 00001 11110 11100  00010 11110 11100
```

### Arithmetic (Shunting‑Yard evaluator)
```
-123 * -8175 = 1,005,525
→ 10001 10010 10011 11110 01100 11000 10001 10111 10101 11110
```

### Scale annotation (decimal storage)
```
-1.234 → -1234 S 3
→ 10001 10010 10011 10100 11110 11011 00011 11110
```
The database stores pure integers. `S` tells the application layer "this integer has N implied decimal places." Decimal arithmetic (scale alignment) happens at the application layer — exactly how financial systems store currency as cents.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  APPLICATION LAYER                       │
│  DecimalArithmetic  │  Geometric Queries  │  Consumers   │
├─────────────────────────────────────────────────────────┤
│                    PARSER LAYER                          │
│  State Machine (NUM ⇄ WORD)  │  Arithmetic Evaluator    │
├─────────────────────────────────────────────────────────┤
│                    CORE LAYER                            │
│  Encoder  │  32-Token Lexicon  │  Checksum (mod 32)     │
├─────────────────────────────────────────────────────────┤
│                   STORAGE LAYER                          │
│  BinaryGrid (append‑only)  │  5‑bit ↔ 8‑bit Packing     │
│  Bit‑addressable  │  O(1) seek/read                     │
└─────────────────────────────────────────────────────────┘
```

**Key design principles:**
- **Determinism**: Same input → same bit pattern, every language, every platform
- **Positional semantics**: Dimension N of a record vector IS the meaning — no field names on disk
- **Geometry over SQL**: `manhattan(record, target) < threshold` replaces `WHERE` clauses
- **Explicit boundaries**: `RECORD` tokens demarcate tuples; `CHECKSUM` tokens provide integrity lighthouses
- **Zero‑schema**: Consumers decide what tokens mean — expression, tuple, word, or anything else

## Project Structure

```
griddb/
├── README.md
├── python/
│   ├── binary_grid_db.py          # Core engine (~750 lines)
│   ├── test_binary_grid_db.py     # 168 tests
│   └── requirements.txt
├── typescript/
│   ├── src/
│   │   ├── types.ts               # Token enum, interfaces, union types
│   │   ├── tokens.ts              # Bidirectional lookup tables
│   │   ├── encoder.ts             # Integer/word/expression/record → tokens
│   │   ├── parser.ts              # FINITE STATE MACHINE (NUM ⇄ WORD)
│   │   ├── checksum.ts            # Modulo‑32 integrity
│   │   ├── serialization.ts       # 5‑bit ↔ 8‑bit packing
│   │   ├── arithmetic.ts          # Shunting‑Yard + DecimalArithmetic
│   │   ├── grid.ts                # Append‑only bit‑addressable storage
│   │   ├── geometry.ts            # Hamming & Manhattan distance queries
│   │   └── index.ts               # Public API
│   ├── tests/
│   │   └── grid.test.ts           # 50+ TypeScript tests
│   ├── package.json
│   └── tsconfig.json
└── examples/
    ├── griddb_explorer.py         # Interactive terminal demo (6 demos)
    └── grid_transformer.py        # Transformer queries the grid via attention
```

## Getting Started

### Python

```bash
cd python
python3 binary_grid_db.py          # Run the demo
python3 -m unittest test_binary_grid_db -v  # Run 168 tests
```

```python
from binary_grid_db import BinaryGridDB, Encoder, manhattan_distance

db = BinaryGridDB()

# Insert records (no schema required)
db.insert_record("NYC", 4071, -7400)   # 40.71, -74.00
db.insert_record("LA",  3405, -11824)  # 34.05, -118.24

# Query by geometric proximity — no SQL, no index
nearby = db.query_manhattan([4071, -7400], max_distance=500)
for r in nearby:
    print(r.value_vector)  # Cities near NYC
```

### TypeScript

```bash
cd typescript
npm install
npm run build
npm test
```

```typescript
import { BinaryGrid, Encoder, queryByManhattan } from 'griddb';

const grid = new BinaryGrid();

// Insert records
grid.appendRecord(Encoder.encodeRecord('NYC', 4071, -7400));
grid.appendRecord(Encoder.encodeRecord('LA', 3405, -11824));

// Geometric query — finds nearby cities without SQL
const results = queryByManhattan(grid, [4071, -7400], 500);
```

### Explore

```bash
# Full interactive demo (Python)
python3 examples/griddb_explorer.py

# Transformer demo — a transformer that queries the grid
python3 examples/grid_transformer.py
```

## Query Model: Geometry over SQL

Traditional databases require indexes, query planners, and foreign keys to answer:

```sql
SELECT * FROM cities
WHERE ABS(lat - 40.71) + ABS(lon + 74.00) < 5.00;
```

GridDB answers with pure integer arithmetic on record vectors:

```
manhattan([4071, -7400], [3995, -7516]) = 192  → PHL is 1.92° from NYC
manhattan([4071, -7400], [4188, -8763]) = 1480 → CHI is 14.8° away
```

**No parser. No planner. No index. Just integers and POPCNT.**

## Key Properties

| Property | PostgreSQL | MongoDB | GridDB |
|---|---|---|---|
| Deterministic encoding | ✗ (version‑dependent) | ✗ (driver‑dependent) | ✓ |
| Schema‑free | ✗ (DDL required) | ✓ | ✓ (positional semantics) |
| Field‑name overhead | ~4 bytes/row (catalog) | ~10‑30 bytes/doc | **0 bytes** |
| Geometry queries | PostGIS extension | 2dsphere index | **Native** |
| Content‑addressable | ✗ | ✗ | ✓ (SHA‑256 over segments) |
| Min query time | O(log n) index scan | O(log n) index scan | O(n) integer scan |
| Transformer‑native input | ✗ | ✗ | ✓ (32‑token vocab) |

## License

MIT

---

*"This is not a tribute to the past; it is a blueprint for a new, minimalist future of data persistence."*
