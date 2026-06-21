#!/usr/bin/env python3
"""
GridDB Secondary Indexes — Hash + B-tree
==========================================
HashIndex:  O(1) equality lookups.  hash(key) → row → chain → record_id.
BTreeIndex: O(log n) range queries.  Tree nodes stored as grid records.

Both are secondary structures over AllocGrid.
They map search keys → record_ids.
The consumer reads the actual record from the primary grid.

Pattern:
  idx = HashIndex("email")
  idx.put("alice@demo.com", record_id=0)
  idx.get("alice@demo.com")  → 0  (O(1))

  btree = BTreeIndex("age")
  btree.put(25, record_id=0)
  btree.put(30, record_id=1)
  btree.range_scan(21, 35)  → [0, 1]  (O(log n + k))
"""

import os
import struct
import hashlib
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass

from binary_grid_db import (
    Token, Encoder, Parser, ParsedNumber, ParsedWord,
    pack_to_bytes, unpack_from_bytes,
)

# Import AllocGrid for backing storage
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from griddb_alloc import AllocGrid, AllocRecord


# ═══════════════════════════════════════════════════════════════════════════════
# Hash Index — O(1) equality lookup
# ═══════════════════════════════════════════════════════════════════════════════

class HashIndex:
    """Hash index: hash(key) → bucket → chain → record_id.

    Each bucket row in the AllocGrid stores a chain of entries:
      WORD(key) NUM(record_id) RECORD

    Collisions resolve by chaining — multiple entries in the same bucket row.
    Deletes use tombstones.

    Usage:
      idx = HashIndex("email_index", buckets=100000)
      idx.put("alice@demo.com", record_id=0)
      idx.get("alice@demo.com")          → 0
      idx.delete("alice@demo.com")
      idx.get("alice@demo.com")          → None
    """

    def __init__(self, name: str, data_dir: str = "./data", buckets: int = 100000):
        self.name = name
        self.buckets = buckets
        index_dir = os.path.join(data_dir, f"idx_hash_{name}")
        self.grid = AllocGrid(data_dir=index_dir)
        self._stats = {'puts': 0, 'gets': 0, 'deletes': 0, 'collisions': 0}

    def _hash(self, key: str) -> int:
        """Hash a key to a bucket number. Deterministic across all languages."""
        h = hashlib.sha256(key.encode()).digest()
        # Take first 8 bytes as uint64, mod buckets
        val = struct.unpack('>Q', h[:8])[0]
        return val % self.buckets

    def put(self, key: str, record_id: int):
        """Insert a (key → record_id) mapping. O(1) amortized."""
        bucket = self._hash(key)

        # Read existing chain
        existing = self.grid.read(bucket)
        chain_tokens = list(existing.tokens) if existing else []

        # Check for duplicate key (update if exists)
        # Walk existing chain entries
        if existing:
            entries = self._parse_chain(existing)
            for i, (ek, erid) in enumerate(entries):
                if ek == key:
                    # Key exists — could update, but for simplicity we
                    # tombstone and append new (append-only semantics)
                    pass

        # Append new chain entry
        new_entry = [
            *Encoder.encode_word(key),
            *Encoder.encode_integer(record_id),
            Token.RECORD,
        ]
        chain_tokens.extend(new_entry)
        self.grid.write(bucket, chain_tokens)
        self._stats['puts'] += 1

    def get(self, key: str) -> Optional[int]:
        """Look up a key. O(1) — hash to bucket, scan chain."""
        bucket = self._hash(key)
        existing = self.grid.read(bucket)
        self._stats['gets'] += 1

        if existing is None or existing.is_tombstone:
            return None

        entries = self._parse_chain(existing)
        for ek, erid in entries:
            if ek == key:
                return erid

        return None

    def delete(self, key: str) -> bool:
        """Remove a key from the index. O(1)."""
        bucket = self._hash(key)
        existing = self.grid.read(bucket)
        self._stats['deletes'] += 1

        if existing is None or existing.is_tombstone:
            return False

        entries = self._parse_chain(existing)
        found = any(ek == key for ek, _ in entries)
        if found:
            # Rebuild chain without the deleted key
            new_chain = []
            for ek, erid in entries:
                if ek != key:
                    new_chain.extend([
                        *Encoder.encode_word(ek),
                        *Encoder.encode_integer(erid),
                        Token.RECORD,
                    ])
            if new_chain:
                self.grid.write(bucket, new_chain)
            else:
                self.grid.delete(bucket)
            return True
        return False

    def _parse_chain(self, record: AllocRecord) -> List[Tuple[str, int]]:
        """Parse a bucket's chain into (key, record_id) pairs.
        Chain format: WORD(key1) NUM(id1) RECORD WORD(key2) NUM(id2) RECORD ...
        Keys may produce multiple WORD tokens (context switching at @, .).
        Join consecutive WORDs, pair with following NUM, skip RECORD.
        """
        entries = []
        parsed = record.parsed
        i = 0
        while i < len(parsed):
            # Collect consecutive WORD tokens (key may be split by context switches)
            if isinstance(parsed[i], ParsedWord):
                key_parts = []
                while i < len(parsed) and isinstance(parsed[i], ParsedWord):
                    key_parts.append(parsed[i].text)
                    i += 1
                key = ''.join(key_parts)
                # Next should be NUM(record_id)
                if i < len(parsed) and isinstance(parsed[i], ParsedNumber):
                    rid = parsed[i].value
                    entries.append((key, rid))
                    i += 1
                    # Skip optional control/RECORD token
                    if i < len(parsed) and hasattr(parsed[i], 'type') and parsed[i].type == 'control':
                        i += 1
                continue
            i += 1
        return entries

    def stats(self) -> dict:
        s = self.grid.stats()
        s.update(self._stats)
        s['buckets'] = self.buckets
        return s

    def close(self):
        self.grid.close()


# ═══════════════════════════════════════════════════════════════════════════════
# B-tree Index — O(log n) range queries
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BTreeNode:
    """A B-tree node stored in the grid."""
    node_id: int
    is_leaf: bool
    keys: List[int]        # sorted integer keys
    children: List[int]    # child node_ids (internal) or record_ids (leaf)
    next_leaf: int         # next leaf node_id (-1 if last)


class BTreeIndex:
    """B-tree index: sorted integer keys → record_ids. O(log n) lookup.

    Each B-tree node is stored as a record in an AllocGrid:
      NUM(node_id) NUM(is_leaf) NUM(num_keys) NUM(next_leaf)
      [NUM(key) NUM(ptr)] × num_keys
      RECORD

    Internal nodes: ptr = child node_id
    Leaf nodes: ptr = data record_id
    Leaves are linked (next_leaf) for range scans.

    For string keys (emails, names): hash to integer first, or use
    a separate string-key B-tree that encodes chars as ASCII ordinals.

    Min degree = 2 (each node has 2-4 keys, root can have 1).

    Usage:
      btree = BTreeIndex("age_index")
      btree.put(25, record_id=0)
      btree.put(30, record_id=1)
      btree.get(25)                    → 0
      btree.range_scan(21, 35)         → [0, 1]  (keys 21 ≤ k < 35)
    """

    MIN_DEGREE = 2  # t=2, nodes hold 2-4 keys (except root)

    def __init__(self, name: str, data_dir: str = "./data"):
        self.name = name
        index_dir = os.path.join(data_dir, f"idx_btree_{name}")
        self.grid = AllocGrid(data_dir=index_dir)
        self._next_node_id = 0
        self._root_id = -1

        # Initialize or load
        self._bootstrap()

    def _bootstrap(self):
        """Load existing tree or create empty."""
        # Root is always at node_id 0. Check if it exists.
        root_rec = self.grid.read(0)
        if root_rec is not None and not root_rec.is_tombstone:
            root_node = self._parse_node(root_rec)
            if root_node:
                self._root_id = 0
                # Find max node_id
                total = self.grid.total_entries
                for nid in range(total):
                    if self.grid.read(nid) is not None:
                        self._next_node_id = max(self._next_node_id, nid + 1)

    # ── Node serialization ───────────────────────────────────────────────

    def _encode_node(self, node: BTreeNode) -> List[Token]:
        """Encode a B-tree node to tokens.
        Layout: ID, is_leaf, n_keys, next,
                [key0, child0, key1, child1, ...]
                [child_last]  ← extra child (K+1 children for K keys)
        """
        tokens = [
            *Encoder.encode_integer(node.node_id),
            *Encoder.encode_integer(1 if node.is_leaf else 0),
            *Encoder.encode_integer(len(node.keys)),
            *Encoder.encode_integer(node.next_leaf),
        ]
        # Interleave keys and children: key0, child0, key1, child1, ...
        for i, key in enumerate(node.keys):
            tokens.extend(Encoder.encode_integer(key))
            tokens.extend(Encoder.encode_integer(
                node.children[i] if i < len(node.children) else -1
            ))
        # Last child (K keys → K+1 children for internal nodes)
        if len(node.children) > len(node.keys):
            tokens.extend(Encoder.encode_integer(node.children[-1]))
        tokens.append(Token.RECORD)
        return tokens

    def _parse_node(self, record: AllocRecord) -> Optional[BTreeNode]:
        """Parse a grid record into a B-tree node.
        Layout: id, leaf?, n_keys, next, k0,c0, k1,c1, ..., c_last
        K keys → 2K+1 data values (K keys + K+1 children).
        Keys at even data positions, children at odd + last position.
        """
        nums = [p.value for p in record.parsed if isinstance(p, ParsedNumber)]
        if len(nums) < 4:
            return None

        node_id = nums[0]
        is_leaf = nums[1] == 1
        num_keys = nums[2]
        next_leaf = nums[3]

        data = nums[4:] if len(nums) > 4 else []
        keys = data[0::2]                     # even positions: key0, key1, ...
        children = data[1::2]                  # odd positions: child0, child1, ...
        if len(data) % 2 == 1:                 # extra child appended at end
            children.append(data[-1])

        return BTreeNode(
            node_id=node_id, is_leaf=bool(is_leaf),
            keys=keys, children=children, next_leaf=next_leaf,
        )

    def _write_node(self, node: BTreeNode):
        """Write a node to the grid."""
        tokens = self._encode_node(node)
        self.grid.write(node.node_id, tokens)
        if node.node_id >= self._next_node_id:
            self._next_node_id = node.node_id + 1

    def _read_node(self, node_id: int) -> Optional[BTreeNode]:
        """Read a node from the grid."""
        rec = self.grid.read(node_id)
        if rec is None or rec.is_tombstone:
            return None
        return self._parse_node(rec)

    def _alloc_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    # ── Core operations ──────────────────────────────────────────────────

    def put(self, key: str, record_id: int):
        """Insert a key → record_id mapping."""
        if self._root_id < 0:
            # First insert: create root leaf
            root = BTreeNode(
                node_id=self._alloc_node_id(),
                is_leaf=True,
                keys=[key],
                children=[record_id],
                next_leaf=-1,
            )
            self._write_node(root)
            self._root_id = root.node_id
            return

        # Walk to leaf
        path = self._find_leaf(key)  # returns (leaf_node, [ancestors])
        if path is None:
            return

        leaf, ancestors = path

        # Insert into leaf
        self._insert_into_leaf(leaf, key, record_id, ancestors)

    def get(self, key: str) -> Optional[int]:
        """Look up a key. O(log n)."""
        if self._root_id < 0:
            return None

        node = self._read_node(self._root_id)
        while node:
            if node.is_leaf:
                # Search leaf keys
                for i, k in enumerate(node.keys):
                    if k == key:
                        return node.children[i]
                return None
            else:
                # Internal node: find child
                child_idx = 0
                for i, k in enumerate(node.keys):
                    if key < k:
                        break
                    child_idx = i + 1
                if child_idx >= len(node.children):
                    child_idx = len(node.children) - 1
                node = self._read_node(node.children[child_idx])

        return None

    def delete(self, key: str) -> bool:
        """Delete a key. O(log n). Returns True if deleted."""
        if self._root_id < 0:
            return False
        # Simplified: tombstone the entry by marking record_id = -1
        # Full B-tree delete is complex (rebalancing). For now, soft delete.
        return self._soft_delete(key)

    def _soft_delete(self, key: int) -> bool:
        """Soft delete: mark the entry's record_id as -1."""
        node = self._read_node(self._root_id)
        while node:
            if node.is_leaf:
                for i, k in enumerate(node.keys):
                    if k == key:
                        node.children[i] = -1
                        self._write_node(node)
                        return True
                return False
            else:
                child_idx = 0
                for i, k in enumerate(node.keys):
                    if key < k:
                        break
                    child_idx = i + 1
                if child_idx >= len(node.children):
                    child_idx = len(node.children) - 1
                node = self._read_node(node.children[child_idx])
        return False

    def range_scan(self, start_key: int, end_key: int) -> List[int]:
        """Find all record_ids with keys in [start_key, end_key). O(log n + k)."""
        results = []
        if self._root_id < 0:
            return results

        # Find the leaf containing start_key
        node = self._read_node(self._root_id)
        while node and not node.is_leaf:
            child_idx = 0
            for i, k in enumerate(node.keys):
                if start_key < k:
                    break
                child_idx = i + 1
            if child_idx >= len(node.children):
                child_idx = len(node.children) - 1
            node = self._read_node(node.children[child_idx])

        # Walk leaf chain
        while node:
            for i, k in enumerate(node.keys):
                if k >= start_key and k < end_key:
                    rid = node.children[i] if i < len(node.children) else -1
                    if rid >= 0:
                        results.append(rid)
                if k >= end_key:
                    return results
            # Next leaf
            if node.next_leaf >= 0:
                node = self._read_node(node.next_leaf)
            else:
                break

        return results

    # ── B-tree internals ─────────────────────────────────────────────────

    def _find_leaf(self, key: int) -> Optional[Tuple[BTreeNode, List[BTreeNode]]]:
        """Walk from root to leaf. Returns (leaf_node, [ancestors])."""
        if self._root_id < 0:
            return None

        ancestors = []
        node = self._read_node(self._root_id)
        if node is None:
            return None

        while node and not node.is_leaf:
            ancestors.append(node)
            child_idx = 0
            for i, k in enumerate(node.keys):
                if key < k:
                    break
                child_idx = i + 1
            if child_idx >= len(node.children):
                child_idx = len(node.children) - 1
            next_node = self._read_node(node.children[child_idx])
            if next_node is None:
                break
            node = next_node

        return (node, ancestors)

    def _insert_into_leaf(self, leaf: BTreeNode, key: int, record_id: int,
                          ancestors: List[BTreeNode]):
        """Insert into a leaf, splitting if necessary."""
        # Check for duplicate
        for i, k in enumerate(leaf.keys):
            if k == key:
                leaf.children[i] = record_id  # Update
                self._write_node(leaf)
                return

        # Insert in sorted order
        insert_idx = 0
        for i, k in enumerate(leaf.keys):
            if key > k:
                insert_idx = i + 1
            else:
                break

        leaf.keys.insert(insert_idx, key)
        leaf.children.insert(insert_idx, record_id)
        self._write_node(leaf)

        # Split if overflow
        max_keys = 2 * self.MIN_DEGREE  # 4 keys max for t=2
        if len(leaf.keys) > max_keys:
            self._split_leaf(leaf, ancestors)

    def _split_leaf(self, leaf: BTreeNode, ancestors: List[BTreeNode]):
        """Split a full leaf node."""
        mid = len(leaf.keys) // 2

        # Right half → new leaf
        new_leaf = BTreeNode(
            node_id=self._alloc_node_id(),
            is_leaf=True,
            keys=leaf.keys[mid:],
            children=leaf.children[mid:],
            next_leaf=leaf.next_leaf,
        )
        self._write_node(new_leaf)

        # Update left leaf
        leaf.keys = leaf.keys[:mid]
        leaf.children = leaf.children[:mid]
        leaf.next_leaf = new_leaf.node_id
        self._write_node(leaf)

        # Promote middle key to parent
        promote_key = new_leaf.keys[0]

        if not ancestors:
            # Create new root
            new_root = BTreeNode(
                node_id=self._alloc_node_id(),
                is_leaf=False,
                keys=[promote_key],
                children=[leaf.node_id, new_leaf.node_id],
                next_leaf=-1,
            )
            self._write_node(new_root)
            self._root_id = new_root.node_id
        else:
            parent = ancestors[-1]
            self._insert_into_internal(parent, promote_key, leaf.node_id,
                                        new_leaf.node_id, ancestors[:-1])

    def _insert_into_internal(self, parent: BTreeNode, key: int,
                               left_child: int, right_child: int,
                               ancestors: List[BTreeNode]):
        """Insert a promoted key + right child into an internal node.
        The left_child should already exist in parent.children (it's the
        child that was split).  We insert the key and the new right child
        right after the left_child's position.
        """
        # Find where left_child is in the parent's children
        pos = -1
        for i, c in enumerate(parent.children):
            if c == left_child:
                pos = i
                break
        if pos < 0:
            # left_child not found — fallback to sorted position
            pos = 0
            for i, k in enumerate(parent.keys):
                if key > k:
                    pos = i + 1
                else:
                    break

        # Insert key at pos, right_child at pos+1
        parent.keys.insert(pos, key)
        parent.children.insert(pos + 1, right_child)
        self._write_node(parent)

        # Split if overflow
        max_keys = 2 * self.MIN_DEGREE
        if len(parent.keys) > max_keys:
            self._split_internal(parent, ancestors)

    def _split_internal(self, node: BTreeNode, ancestors: List[BTreeNode]):
        """Split a full internal node."""
        mid = len(node.keys) // 2
        promote_key = node.keys[mid]

        # Right half → new node (keys AFTER mid, children AFTER mid+1)
        new_node = BTreeNode(
            node_id=self._alloc_node_id(),
            is_leaf=False,
            keys=node.keys[mid + 1:],
            children=node.children[mid + 1:],
            next_leaf=-1,
        )
        self._write_node(new_node)

        # Left half stays (keys before mid, children up to mid+1)
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        self._write_node(node)

        if not ancestors:
            # Create new root
            new_root = BTreeNode(
                node_id=self._alloc_node_id(),
                is_leaf=False,
                keys=[promote_key],
                children=[node.node_id, new_node.node_id],
                next_leaf=-1,
            )
            self._write_node(new_root)
            self._root_id = new_root.node_id
        else:
            # Promote to parent
            self._insert_into_internal(
                ancestors[-1], promote_key,
                node.node_id, new_node.node_id,
                ancestors[:-1]
            )

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        s = self.grid.stats()
        s['root_id'] = self._root_id
        s['next_node_id'] = self._next_node_id
        return s

    def close(self):
        self.grid.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import tempfile, shutil, time

    demo_dir = tempfile.mkdtemp(prefix='griddb_index_demo_')
    print(f"Demo dir: {demo_dir}")
    print("═" * 60)
    print("  GridDB Secondary Indexes — Hash + B-tree")
    print("═" * 60)

    try:
        # ═════════════════════════════════════════════════════════════════
        # Hash Index Demo
        # ═════════════════════════════════════════════════════════════════
        print("\n── Hash Index ──")

        idx = HashIndex("email", data_dir=demo_dir, buckets=1000)

        # Insert mappings
        users = [
            ("alice@demo.com", 0),
            ("bob@demo.com", 1),
            ("charlie@demo.com", 42),
            ("diana@demo.com", 99),
        ]
        for email, rid in users:
            idx.put(email, rid)
        print(f"  Inserted {len(users)} email → record_id mappings")

        # Lookup
        for email, expected in users:
            result = idx.get(email)
            status = "✓" if result == expected else "✗"
            print(f"  get('{email}') → {result} {status}")

        # Non-existent
        missing = idx.get("nonexistent@demo.com")
        print(f"  get('nonexistent@demo.com') → {missing} (expected None)")

        # Delete
        idx.delete("bob@demo.com")
        deleted = idx.get("bob@demo.com")
        print(f"  delete('bob@demo.com') → get → {deleted} (expected None)")

        hs = idx.stats()
        print(f"  Stats: {hs['buckets']} buckets, "
              f"{hs['puts']} puts, {hs['gets']} gets, {hs['deletes']} deletes")

        # ═════════════════════════════════════════════════════════════════
        # B-tree Index Demo
        # ═════════════════════════════════════════════════════════════════
        print("\n── B-tree Index ──")

        btree = BTreeIndex("age", data_dir=demo_dir)

        # Insert age → record_id mappings (integer keys)
        ages = [(18, 0), (25, 1), (30, 2), (22, 3), (27, 4),
                (35, 5), (20, 6), (40, 7), (15, 8), (33, 9),
                (28, 10), (45, 11), (19, 12), (50, 13), (21, 14)]

        for age, rid in ages:
            btree.put(age, rid)
        print(f"  Inserted {len(ages)} age → record_id mappings")

        # Point lookups
        for age, expected in [(25, 1), (40, 7), (15, 8)]:
            result = btree.get(age)
            status = "✓" if result == expected else "✗"
            print(f"  get({age}) → record_id={result} {status}")

        # Range scan
        print(f"\n  Range scan [20, 35):")
        results = btree.range_scan(20, 35)
        print(f"  Records with age 20-34: {results}")
        print(f"  Expected: [6, 14, 3, 1, 4, 10, 2, 9] (ages 20,21,22,25,27,28,30,33)")

        # Count
        print(f"\n  B-tree stats: {btree.stats()}")

        # O(log n) benchmark
        print(f"\n── O(log n) Benchmark ──")
        # Build larger tree
        big_btree = BTreeIndex("big", data_dir=demo_dir)
        for i in range(1000):
            big_btree.put(i * 10, i)

        # Time lookups
        for key in [0, 5000, 9990]:
            start = time.perf_counter()
            result = big_btree.get(key)
            elapsed = (time.perf_counter() - start) * 1_000_000
            print(f"  get({key}) → {result} in {elapsed:.1f}µs")

        # Range scan benchmark
        start = time.perf_counter()
        range_results = big_btree.range_scan(200, 300)
        range_time = (time.perf_counter() - start) * 1_000_000
        print(f"  range_scan('00200','00300') → {len(range_results)} results in {range_time:.1f}µs")

        idx.close()
        btree.close()
        big_btree.close()

        print("\n" + "═" * 60)
        print("  Index demo complete")
        print("═" * 60)

    finally:
        shutil.rmtree(demo_dir, ignore_errors=True)
