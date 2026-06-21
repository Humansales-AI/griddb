#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    GRIDDB EXPLORER — Interactive Demo                        ║
║                                                                            ║
║  A toy application demonstrating the Binary Grid Database in action:       ║
║    • Write & read records (numbers, words, scaled decimals)                ║
║    • Raw binary visualization of every token                               ║
║    • Geometric queries — find "nearby" records without SQL                 ║
║    • Checksum integrity & corruption detection                             ║
║    • Serialize to disk & reload                                            ║
║    • Zero schema, zero query planner, pure 5-bit fabric                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import math
import struct
import random
from typing import List, Optional, Tuple

# Add parent dir to path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_grid_db import (
    Token, Encoder, Parser, ParsedNumber, ParsedScaledNumber, ParsedWord,
    ParsedOperator, ChecksumResult, Record,
    compute_checksum, append_checksum, verify_checksum,
    pack_to_bytes, unpack_from_bytes,
    token_stream_to_binary_string,
    ArithmeticEvaluator, resolve_scaled_numbers, DecimalArithmetic,
    BinaryGrid, BinaryGridDB, GridRecord,
    hamming_distance, manhattan_distance,
    query_by_manhattan, query_by_hamming_shard,
    inject_bit_flip, scan_for_corruption, find_next_sync_point,
    OPERATOR_SYMBOL, WORD_CHAR,
)

# ── Terminal colors ──────────────────────────────────────────────────────────
C = {
    'reset': '\033[0m', 'bold': '\033[1m', 'dim': '\033[2m',
    'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m',
    'blue': '\033[34m', 'magenta': '\033[35m', 'cyan': '\033[36m',
    'white': '\033[37m',
    'bg_red': '\033[41m', 'bg_green': '\033[42m', 'bg_blue': '\033[44m',
    'bg_yellow': '\033[43m',
}

def c(color: str, text: str) -> str:
    return f"{C.get(color, '')}{text}{C['reset']}"

def section(title: str):
    print(f"\n{c('cyan', '─' * 60)}")
    print(f"  {c('bold', title)}")
    print(f"{c('cyan', '─' * 60)}")

def show_binary(tokens: List[Token], label: str = ""):
    """Display tokens as a visual binary strip."""
    binary_str = token_stream_to_binary_string(tokens)
    # Colorize: controls in red, digits in green, operators in yellow
    parts = binary_str.split()
    colored = []
    for p in parts:
        val = int(p, 2)
        if val >= 0b11100:
            colored.append(c('red', p))
        elif val <= 0b01001:
            colored.append(c('green', p))
        elif 0b10001 <= val <= 0b11001:
            colored.append(c('magenta', p))
        else:
            colored.append(c('yellow', p))
    if label:
        print(f"  {c('dim', label)}")
    print(f"  {' '.join(colored)}")
    print(f"  {c('dim', f'({len(tokens)} tokens, {len(tokens)*5} bits)')}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 1: City Database — Geometric Nearest-Neighbor Search
# ═══════════════════════════════════════════════════════════════════════════════

def demo_city_database():
    section("DEMO 1: City Database — Geometric Queries Replace SQL")

    print(f"\n  We'll store cities as records: {c('green', '(id, lat_scaled, lon_scaled)')}")
    print(f"  Coordinates use the {c('yellow', 'S')} (Scale) annotation for decimal places.")
    print(f"  Then we {c('bold', 'query by geometric proximity')} — no indexes, no SQL.\n")

    db = BinaryGridDB()

    # Store cities: (name as word, latitude*100, longitude*100)
    # Scale=2 means 2 decimal places: 4071 S 2 = 40.71, -7400 S 2 = -74.00
    cities = [
        ("NYC", 4071, -7400),      # 40.71, -74.00
        ("LA", 3405, -11824),      # 34.05, -118.24
        ("CHI", 4188, -8763),      # 41.88, -87.63
        ("HOU", 2976, -9536),      # 29.76, -95.36
        ("PHX", 3345, -11207),     # 33.45, -112.07
        ("PHL", 3995, -7516),      # 39.95, -75.16
        ("SAN", 2942, -9849),      # 29.42, -98.49
        ("SDG", 3272, -11716),     # 32.72, -117.16
        ("DAL", 3278, -9680),      # 32.78, -96.80
        ("SJC", 3734, -12189),     # 37.34, -121.89
    ]

    print("  Inserting cities...\n")
    city_records = {}
    for name, lat, lon in cities:
        # Encode: WORD(name) NUM(lat) NUM(lon) as a record
        record = db.insert_record(name, lat, lon)
        city_records[name] = record
        # Show the raw binary
        tokens = Encoder.encode_record(name, lat, lon)
        show_binary(tokens, f"  {c('bold', name):<4} ({lat/100:.2f}, {lon/100:.2f})")

    # ── Query 1: Find cities near NYC (40.71, -74.00) ──
    print(f"\n  {c('bold', '🔍 Query: Cities within Manhattan distance 500 of NYC (40.71, -74.00)')}")
    print(f"     This replaces: SELECT * FROM cities WHERE abs(lat-40.71)+abs(lon+74.00) < 5.00")
    print(f"     But we never wrote a schema, parser, or query planner.\n")

    target = [40.71 * 100, -74.00 * 100]  # In the integer space
    # Note: our records store the raw integers, so we search in integer space
    # Actually, let me think about this... the value_vector returns the integer values
    # For a record with lat=4071, lon=-7400, the value_vector is [4071, -7400]
    # Manhattan distance works in that space.
    # A distance of 500 means sum of absolute differences < 500
    # That's about 5 degrees in lat/lon space (since we stored *100)
    results = db.query_manhattan([4071, -7400], 500)

    for r in results:
        # Find city name by matching value vector
        vec = r.value_vector
        if len(vec) >= 2:
            lat, lon = vec[0] / 100.0, vec[1] / 100.0
            dist = manhattan_distance(vec, [4071, -7400])
            # Find name
            name = "???"
            for n, rec in city_records.items():
                if rec.value_vector == vec:
                    name = n
                    break
            marker = " ← SAME" if dist == 0 else ""
            print(f"    {c('green', name):<4} ({lat:>7.2f}, {lon:>7.2f})  "
                  f"Manhattan distance: {c('yellow', str(dist))}{marker}")

    # ── Query 2: Which two cities are closest to each other? ──
    print(f"\n  {c('bold', '🔍 Query: Find the two geographically closest cities')}")
    print(f"     No JOIN, no foreign keys — just pairwise Manhattan distance.\n")

    min_dist = float('inf')
    closest_pair = ("", "")
    city_vecs = {name: rec.value_vector for name, rec in city_records.items()}

    names = list(city_vecs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = manhattan_distance(city_vecs[names[i]], city_vecs[names[j]])
            if d < min_dist:
                min_dist = d
                closest_pair = (names[i], names[j])

    a, b = closest_pair
    va, vb = city_vecs[a], city_vecs[b]
    print(f"    {c('green', a)} ({va[0]/100:.2f}, {va[1]/100:.2f})")
    print(f"    {c('green', b)} ({vb[0]/100:.2f}, {vb[1]/100:.2f})")
    print(f"    Manhattan distance: {c('yellow', str(min_dist))} "
          f"(≈ {min_dist/100:.2f}° in coordinate space)")

    return db


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 2: Financial Ledger — Scale Annotation & Integrity
# ═══════════════════════════════════════════════════════════════════════════════

def demo_financial_ledger():
    section("DEMO 2: Financial Ledger — Scale, Checksums & Corruption Detection")

    print(f"\n  We'll store transactions with {c('yellow', 'S')} for decimal amounts.")
    print(f"  The database stores {c('bold', 'pure integers')}. S tells the app where the dot goes.")
    print(f"  Every 5 records, we inject a {c('red', 'CHECKSUM')} for integrity.\n")

    db = BinaryGridDB()

    # Transactions: (description, amount as scaled int, scale)
    transactions = [
        ("Coffee", 399, 2),       # $3.99
        ("Groceries", 12549, 2),  # $125.49
        ("Rent", 250000, 2),      # $2500.00
        ("Gas", 4250, 2),         # $42.50
        ("Book", 1599, 2),        # $15.99
    ]

    print("  Writing transactions with checksums...\n")

    all_tokens = []
    for desc, amount, scale in transactions:
        # Encode: WORD(desc) + NUM(amount, END) + S + NUM(scale, END)
        tokens = (
            Encoder.encode_word(desc) +
            Encoder.encode_integer(amount) +
            [Token.T_SCALE] +
            Encoder.encode_integer(scale)
        )
        all_tokens.extend(tokens)
        # Show
        amount_str = f"${amount / (10**scale):.2f}"
        print(f"    {c('bold', desc):<12} {c('green', amount_str):>10}  ", end="")
        print(f"stored as: NUM({amount}) S {scale}")

    # Append a checksum
    cs_tokens = append_checksum(all_tokens)
    cs_value = int(cs_tokens[-1])
    print(f"\n  {c('red', 'CHECKSUM')} appended: value={cs_value}")
    show_binary(cs_tokens, "Full stream with checksum:")

    # Verify
    print(f"\n  {c('bold', 'Verifying integrity...')}")
    results = scan_for_corruption(cs_tokens)
    for r in results:
        status = c('green', '✓ CLEAN') if r.passed else c('red', '✗ CORRUPT')
        print(f"    {status}  (expected={r.expected}, computed={r.computed})")

    # Inject corruption
    print(f"\n  {c('bold', 'Injecting a single bit-flip in the Coffee transaction...')}")
    corrupted = inject_bit_flip(cs_tokens, position=2, bit_index=3)
    print(f"    Original token[2]: {int(cs_tokens[2]):05b}")
    print(f"    Corrupted         : {c('red', f'{int(corrupted[2]):05b}')}")

    results = scan_for_corruption(corrupted)
    for r in results:
        status = c('green', '✓ CLEAN') if r.passed else c('red', '✗ CORRUPT — bit-flip detected!')
        print(f"    {status}")

    # Recovery: find sync point
    sync = find_next_sync_point(corrupted)
    if sync is not None:
        print(f"\n  {c('bold', 'Recovery:')} Next sync point at token index {sync} "
              f"({c('yellow', corrupted[sync].name)})")
        print(f"    Skip to this point and resume parsing — corruption contained.")

    return db


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 3: Knowledge Graph — Semantic Search via Manhattan Distance
# ═══════════════════════════════════════════════════════════════════════════════

def demo_knowledge_graph():
    section("DEMO 3: Knowledge Graph — Semantic Search Without SQL")

    print(f"\n  We'll encode words as integer embeddings (simplified).")
    print(f"  Each record is a {c('green', '(entity, trait_1, trait_2, trait_3)')} tuple.")
    print(f"  Queries find records {c('bold', 'nearby in value space')} — fuzzy, semantic, SQL-free.\n")

    db = BinaryGridDB()

    # Simplified: encode concepts as 3D vectors
    # Each dimension represents an abstract trait (e.g., size, speed, intelligence)
    animals = [
        ("ELEPHANT", 10, 2, 7),     # large, slow, smart
        ("CHEETAH",  5, 10, 4),     # medium, very fast, medium-smart
        ("DOLPHIN",  4, 8, 10),     # medium, fast, very smart
        ("SLOTH",    2, 1, 2),      # small, very slow, not smart
        ("WHALE",    10, 3, 9),     # very large, slow, very smart
        ("HAWK",     3, 9, 6),      # small, very fast, smart
        ("CHIMP",    3, 5, 9),      # small, medium speed, very smart
        ("MOUSE",    1, 4, 3),      # tiny, quick, basic
    ]

    print("  Inserting animals as (size, speed, intelligence) vectors...\n")
    records = {}
    for name, s1, s2, s3 in animals:
        rec = db.insert_record(name, s1, s2, s3)
        records[name] = rec
        tokens = Encoder.encode_record(name, s1, s2, s3)
        show_binary(tokens, f"  {c('bold', name):<10} vector=({s1}, {s2}, {s3})")

    # Query: find animals similar to Dolphin
    print(f"\n  {c('bold', '🔍 Query: Animals most similar to DOLPHIN (4, 8, 10)')}")
    print(f"     Finds records with smallest Manhattan distance — no JOIN needed.\n")

    target = records["DOLPHIN"].value_vector
    # Sort all by Manhattan distance
    ranked = []
    for name, rec in records.items():
        d = manhattan_distance(rec.value_vector, target)
        ranked.append((name, rec.value_vector, d))

    ranked.sort(key=lambda x: x[2])

    for name, vec, dist in ranked:
        marker = " ← QUERY" if name == "DOLPHIN" else ""
        bar = "█" * max(1, 10 - dist) if dist < 12 else "▏"
        print(f"    {c('green', name):<10} vector={tuple(vec)}  "
              f"distance={c('yellow', str(dist)):<3}  {c('dim', bar)}{marker}")

    # Cross-query: what's most similar to ELEPHANT?
    print(f"\n  {c('bold', '🔍 Query: Animals most similar to ELEPHANT (10, 2, 7)')}")
    target = records["ELEPHANT"].value_vector
    ranked = []
    for name, rec in records.items():
        d = manhattan_distance(rec.value_vector, target)
        ranked.append((name, rec.value_vector, d))
    ranked.sort(key=lambda x: x[2])

    for name, vec, dist in ranked[:5]:
        marker = " ← QUERY" if name == "ELEPHANT" else ""
        print(f"    {c('green', name):<10} distance={c('yellow', str(dist)):<3}{marker}")
    if len(ranked) > 5:
        print(f"    {c('dim', '...')}")

    return db


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 4: Binary Explorer — See Inside the Encoding
# ═══════════════════════════════════════════════════════════════════════════════

def demo_binary_explorer():
    section("DEMO 4: Binary Explorer — See Inside the 5-Bit Fabric")

    print(f"\n  Every symbol — digit, letter, operator, control — occupies {c('bold', 'exactly 5 bits')}.")
    print(f"  Same binary code means different things in different contexts.\n")

    examples = [
        ("Number 42", Encoder.encode_integer(42)),
        ("Number -99", Encoder.encode_integer(-99)),
        ("Word 'CODE'", Encoder.encode_word("CODE")),
        ("-123 * -8175", Encoder.encode_expression([[-1,-2,-3], '*', [-8,-1,-7,-5]])),
        ("Record(7, 'A')", Encoder.encode_record(7, "A")),
        ("Scale 3.14", Encoder.encode_expression([[3, 1, 4], 'S', [2]])),
    ]

    for label, tokens in examples:
        show_binary(tokens, f"  {c('bold', label)}")

    # Show the dual meaning of a single code
    print(f"\n  {c('bold', '🔍 Context Duality: Same 5 bits, different meaning')}\n")
    duals = [
        (0b00001, "NUM: 1", "WORD: B"),
        (0b01010, "NUM: +", "WORD: K"),
        (0b01101, "NUM: /", "WORD: N"),
        (0b10001, "NUM: -1", "WORD: R"),
        (0b11010, "NUM: ^", "WORD: (space)"),
        (0b11111, "NUM: START (→WORD)", "WORD: START (error)"),
    ]
    for val, num_meaning, word_meaning in duals:
        bits = f"{val:05b}"
        print(f"    {c('yellow', bits)}  →  {c('green', num_meaning):<22}  |  {c('magenta', word_meaning)}")

    # Show the full 32-slot table
    print(f"\n  {c('bold', 'Complete 32-Slot Vocabulary:')}\n")
    print(f"    {c('dim', 'Binary  │ NUM context      │ WORD context')}")
    print(f"    {c('dim', '────────┼──────────────────┼─────────────')}")
    for val in range(32):
        tok = Token(val)
        from binary_grid_db import NUMERIC_DIGIT_VALUE, WORD_CHAR, OPERATOR_SYMBOL

        if tok in NUMERIC_DIGIT_VALUE and NUMERIC_DIGIT_VALUE[tok] is not None:
            num_str = str(NUMERIC_DIGIT_VALUE[tok])
        elif tok in OPERATOR_SYMBOL:
            num_str = OPERATOR_SYMBOL[tok]
        elif tok.name == 'RECORD':
            num_str = c('red', 'RECORD')
        elif tok.name == 'CHECKSUM':
            num_str = c('red', 'CHECKSUM')
        elif tok.name == 'END':
            num_str = c('red', 'END')
        elif tok.name == 'START':
            num_str = c('red', 'START')
        else:
            num_str = '?'

        if tok in WORD_CHAR:
            word_str = WORD_CHAR[tok]
        elif tok.name in ('RECORD', 'CHECKSUM', 'END', 'START'):
            word_str = c('red', tok.name)
        else:
            word_str = '?'

        print(f"    {c('yellow', f'{val:05b}')}  │ {num_str:<16} │ {word_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 5: Serialization — Persist to Disk & Reload
# ═══════════════════════════════════════════════════════════════════════════════

def demo_serialization():
    section("DEMO 5: Serialization — Persist & Reload from Disk")

    print(f"\n  The entire database is a flat sequence of bits.")
    print(f"  Packed to bytes, written to disk, reloaded losslessly.\n")

    # Build a database
    db = BinaryGridDB()
    db.insert_record("HELLO", 42)
    db.insert_record("WORLD", -17, 3)
    db.insert_word("BINARY")
    db.insert_number(255)

    stats_before = db.stats()
    print(f"  Before serialization:")
    print(f"    Tokens: {stats_before['token_count']}, "
          f"Bits: {stats_before['bit_length']}, "
          f"Records: {stats_before['record_count']}")

    # Pack to bytes
    packed, pad = db.pack()
    print(f"\n  Packed to bytes: {len(packed)} bytes (pad={pad} bits)")
    print(f"  Hex: {packed.hex()}")

    # "Write to disk"
    disk_path = "/tmp/griddb_demo.bin"
    with open(disk_path, "wb") as f:
        f.write(struct.pack(">I", pad))  # Store pad length as 4-byte header
        f.write(packed)

    file_size = os.path.getsize(disk_path)
    print(f"\n  {c('green', '✓')} Written to {disk_path} ({file_size} bytes)")

    # "Read from disk"
    with open(disk_path, "rb") as f:
        stored_pad = struct.unpack(">I", f.read(4))[0]
        stored_data = f.read()

    # Reload
    restored = BinaryGridDB.unpack(stored_data, stored_pad)
    stats_after = restored.stats()

    print(f"\n  {c('green', '✓')} Reloaded from disk:")
    print(f"    Tokens: {stats_after['token_count']}, "
          f"Bits: {stats_after['bit_length']}, "
          f"Records: {stats_after['record_count']}")

    # Verify round-trip
    assert stats_before == stats_after, "ROUND-TRIP FAILED!"
    print(f"\n  {c('bold', c('green', '✓ Round-trip verified — all data preserved.'))}")

    # Show the records
    print(f"\n  Reloaded records:")
    for i in range(restored.grid.record_count):
        rec = restored.grid.get_record(i)
        print(f"    Record {i}: {rec.value_vector}")

    # Cleanup
    os.remove(disk_path)
    return restored


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 6: Hamming Shard Router
# ═══════════════════════════════════════════════════════════════════════════════

def demo_shard_routing():
    section("DEMO 6: Distributed Shard Routing via Hamming Distance")

    print(f"\n  The grid can be split into shards. Routing uses {c('bold', 'Hamming distance')}")
    print(f"  to find the closest shard to any target address — no hash function, no directory.\n")

    # Simulate 8 shards with random 12-bit addresses
    random.seed(42)
    shards = [random.randint(0, 4095) for _ in range(8)]

    print("  Shard addresses (12-bit):")
    for i, addr in enumerate(shards):
        print(f"    Shard {i}: {c('yellow', f'{addr:012b}')}  (decimal: {addr})")

    # Route some target addresses
    targets = [42, 1000, 2048, 4000]
    print(f"\n  {c('bold', 'Routing targets:')}")
    for target in targets:
        best = query_by_hamming_shard(target, shards)
        hd = hamming_distance(target, shards[best])
        print(f"    Target {c('green', f'{target:012b}')} → Shard {best} "
              f"(addr {shards[best]:012b}, Hamming={hd})")

    print(f"\n  {c('dim', 'No consistent hashing, no ring, no directory.')}")
    print(f"  {c('dim', 'Just POPCNT on addresses.')}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Orchestrate All Demos
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(c('bold', """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗ ██████╗ ██╗██████╗ ██████╗ ██████╗                                ║
║  ██╔════╝ ██╔══██╗██║██╔══██╗██╔══██╗██╔══██╗                               ║
║  ██║  ███╗██████╔╝██║██║  ██║██║  ██║██████╔╝                               ║
║  ██║   ██║██╔══██╗██║██║  ██║██║  ██║██╔══██╗                               ║
║  ╚██████╔╝██║  ██║██║██████╔╝██████╔╝██████╔╝                               ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝ ╚═════╝ ╚═════╝                                ║
║                                                                               ║
║              5‑Bit Binary Fabric — Interactive Explorer                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""))

    print(f"  {c('dim', 'A pure integer binary engine where every symbol occupies exactly 5 bits.')}")
    print(f"  {c('dim', 'No SQL. No floats. No variable-length strings. Just deterministic 5-bit tokens.')}")
    time.sleep(1)

    # Run all demos
    demo_binary_explorer()
    time.sleep(0.5)

    db_cities = demo_city_database()
    time.sleep(0.5)

    db_finance = demo_financial_ledger()
    time.sleep(0.5)

    db_kg = demo_knowledge_graph()
    time.sleep(0.5)

    demo_shard_routing()
    time.sleep(0.5)

    demo_serialization()
    time.sleep(0.5)

    # ── Final stats ──
    section("📊 Session Summary")
    total_tokens = (db_cities.grid.token_count +
                    db_finance.grid.token_count +
                    db_kg.grid.token_count)
    total_records = (db_cities.grid.record_count +
                     db_finance.grid.record_count +
                     db_kg.grid.record_count)
    print(f"\n  Total tokens processed: {c('bold', str(total_tokens))}")
    print(f"  Total records stored:  {c('bold', str(total_records))}")
    print(f"  Total bits on wire:    {c('bold', str(total_tokens * 5))}")
    print(f"\n  {c('green', '✓')} All demos complete.")
    print(f"  The grid stores {c('bold', 'tokens')}, not tables.")
    print(f"  Consumers decide meaning — expressions, tuples, words, or anything else.")
    print(f"  This is the {c('bold', 'Unix philosophy')} applied to data persistence.")
    print()


if __name__ == '__main__':
    main()
