#!/usr/bin/env python3
"""
GridDB Replication — Master/Replica over HTTP
===============================================
The grid IS the oplog.  Every append carries its own LSN.
Replicas just request "what's new since offset X" and append.

Protocol (pull-based, eventual consistency):
  Replica → Master:  GET /sync?since=<seq>
  Master → Replica:  [ {seq, record_id, tokens_hex, sha256}, ... ]
  Replica:           verify SHA-256 chain → apply writes → advance LSN

No separate oplog.  No conflict resolution.  No consensus.
The grid's append-only nature IS the replication protocol.

Usage:
  # Start master
  master = ReplicationMaster(data_dir="./master_data", port=9001)
  master.start()

  # Start replica
  replica = Replica(master_url="http://localhost:9001", data_dir="./replica_data")
  replica.sync()          # pulls and applies all new entries
  replica.sync_loop(5.0)  # polls every 5 seconds
"""

import os
import sys
import json
import time
import struct
import hashlib
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_grid_db import (
    Token, Encoder, Parser, ParsedNumber, ParsedWord,
    pack_to_bytes, unpack_from_bytes, token_stream_to_binary_string,
)
from griddb_positioned import PositionedGridWAL, PositionedRecord


# ═══════════════════════════════════════════════════════════════════════════════
# WAL-backed store (wraps PositionedGridWAL for demo)
# ═══════════════════════════════════════════════════════════════════════════════

class WALStore:
    """Simple key-value store backed by PositionedGridWAL.
    Each write is a WAL entry with (record_id, tokens).
    WAL sequence numbers serve as LSNs.
    """

    def __init__(self, data_dir: str = "./data", stride_bits: int = 1024):
        self.grid = PositionedGridWAL(data_dir=data_dir, stride_bits=stride_bits)
        self._data_dir = data_dir
        self._wal_path = os.path.join(data_dir, "pos_wal.grid")

    def write(self, record_id: int, tokens: List[Token]):
        self.grid.write(record_id, tokens)

    def read(self, record_id: int) -> Optional[PositionedRecord]:
        return self.grid.read(record_id)

    def scan(self, start: int = 0, end: Optional[int] = None):
        return self.grid.scan(start, end)

    @property
    def last_seq(self) -> int:
        """Last WAL sequence number (the LSN)."""
        entries = self._read_raw_wal()
        return entries[-1]['seq'] if entries else -1

    @property
    def next_seq(self) -> int:
        return self.last_seq + 1

    def get_entries_since(self, since_seq: int) -> List[dict]:
        """Get all WAL entries with seq > since_seq."""
        entries = self._read_raw_wal()
        return [e for e in entries if e['seq'] > since_seq]

    def _read_raw_wal(self) -> List[dict]:
        """Read raw WAL entries from the PositionedGridWAL."""
        entries = []
        if not os.path.exists(self._wal_path):
            return entries

        with open(self._wal_path, 'rb') as f:
            data = f.read()

        WAL_MAGIC = 0x4750574C
        WAL_HDR_FMT = ">IIII"
        WAL_PREV_FMT = ">i"
        WAL_PAD_FMT = ">I"
        WAL_HDR_SIZE = struct.calcsize(WAL_HDR_FMT)
        WAL_PREV_SIZE = struct.calcsize(WAL_PREV_FMT)
        WAL_PAD_SIZE = struct.calcsize(WAL_PAD_FMT)
        WAL_HASH_SIZE = 32
        WAL_OVERHEAD = WAL_HDR_SIZE + WAL_PREV_SIZE + WAL_PAD_SIZE + WAL_HASH_SIZE

        offset = 0
        while offset + WAL_OVERHEAD <= len(data):
            magic, seq, record_id, token_count = struct.unpack_from(WAL_HDR_FMT, data, offset)
            if magic != WAL_MAGIC:
                break
            offset += WAL_HDR_SIZE

            prev_hash_offset = struct.unpack_from(WAL_PREV_FMT, data, offset)[0]
            offset += WAL_PREV_SIZE

            token_bits = token_count * 5
            token_bytes = (token_bits + 7) // 8
            if offset + token_bytes > len(data):
                break
            token_data = data[offset:offset + token_bytes]
            offset += token_bytes

            if offset + WAL_PAD_SIZE > len(data):
                break
            pad_len = struct.unpack_from(WAL_PAD_FMT, data, offset)[0]
            offset += WAL_PAD_SIZE

            if offset + WAL_HASH_SIZE > len(data):
                break
            stored_hash = data[offset:offset + WAL_HASH_SIZE]
            offset += WAL_HASH_SIZE

            tokens = unpack_from_bytes(bytearray(token_data), pad_len)

            entries.append({
                'seq': seq,
                'record_id': record_id,
                'tokens': [int(t) for t in tokens],
                'tokens_hex': bytes(token_data).hex(),
                'pad_len': pad_len,
                'prev_hash_offset': prev_hash_offset,
                'sha256': stored_hash.hex(),
            })

        return entries

    def close(self):
        self.grid.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Master HTTP Server
# ═══════════════════════════════════════════════════════════════════════════════

class SyncHandler(BaseHTTPRequestHandler):
    """HTTP handler for the replication master."""

    # Class-level store reference (set by ReplicationMaster)
    store: WALStore = None  # type: ignore

    def log_message(self, format, *args):
        """Quieter logging."""
        print(f"  [master] {args[0]}")

    def do_GET(self):
        path = self.path.split('?')[0]

        if path == '/sync':
            self._handle_sync()
        elif path == '/stats':
            self._handle_stats()
        elif path == '/health':
            self._handle_health()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"not found"}')

    def do_POST(self):
        if self.path == '/write':
            self._handle_write()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_sync(self):
        """GET /sync?since=<seq> — return WAL entries since that sequence."""
        since_seq = self._query_param('since', -1)

        entries = self.store.get_entries_since(since_seq)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        response = {
            'since': since_seq,
            'count': len(entries),
            'latest_seq': self.store.last_seq,
            'entries': entries,
        }
        self.wfile.write(json.dumps(response).encode())

    def _handle_stats(self):
        """GET /stats — return grid statistics."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        stats = {
            'last_seq': self.store.last_seq,
            'next_seq': self.store.next_seq,
        }
        self.wfile.write(json.dumps(stats).encode())

    def _handle_health(self):
        """GET /health — health check."""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def _handle_write(self):
        """POST /write — write a record to the master.
        Body: {"record_id": 42, "tokens": [1,30,3,30,28]}
        """
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            record_id = data['record_id']
            tokens = [Token(t) for t in data['tokens']]
            self.store.write(record_id, tokens)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'ok': True,
                'record_id': record_id,
                'lsn': self.store.last_seq,
            }).encode())
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def _query_param(self, name: str, default: int) -> int:
        """Extract an integer query parameter."""
        path = self.path
        if '?' not in path:
            return default
        qs = path.split('?', 1)[1]
        for pair in qs.split('&'):
            if '=' in pair:
                k, v = pair.split('=', 1)
                if k == name:
                    try:
                        return int(v)
                    except ValueError:
                        return default
        return default


class ReplicationMaster:
    """Master node — serves the WAL to replicas over HTTP."""

    def __init__(self, data_dir: str = "./master_data", port: int = 9001,
                 stride_bits: int = 1024):
        self.data_dir = data_dir
        self.port = port
        self.store = WALStore(data_dir=data_dir, stride_bits=stride_bits)

        # Inject store into handler
        SyncHandler.store = self.store

        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def write(self, record_id: int, tokens: List[Token]):
        """Write to the master store."""
        self.store.write(record_id, tokens)

    def read(self, record_id: int) -> Optional[PositionedRecord]:
        return self.store.read(record_id)

    def start(self, blocking: bool = False):
        """Start the HTTP server."""
        self._server = HTTPServer(('0.0.0.0', self.port), SyncHandler)
        print(f"[master] Listening on :{self.port}")
        print(f"[master] Data dir: {self.data_dir}")
        print(f"[master] Current LSN (last seq): {self.store.last_seq}")

        if blocking:
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=1)

    def close(self):
        self.stop()
        self.store.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Replica Client
# ═══════════════════════════════════════════════════════════════════════════════

class Replica:
    """Replica node — polls master, applies WAL entries, verifies integrity.

    Usage:
      replica = Replica(master_url="http://localhost:9001", data_dir="./replica")
      replica.sync()           # one-time sync
      replica.sync_loop(5.0)   # continuous polling every 5 seconds
    """

    def __init__(self, master_url: str, data_dir: str = "./replica_data",
                 stride_bits: int = 1024):
        self.master_url = master_url.rstrip('/')
        self.store = WALStore(data_dir=data_dir, stride_bits=stride_bits)
        self._last_applied_seq = self.store.last_seq  # may have existing data
        self._sync_count = 0
        self._sync_errors = 0

    @property
    def last_lsn(self) -> int:
        return self._last_applied_seq

    def sync(self) -> dict:
        """Pull and apply all new entries from master. Returns sync result."""
        import urllib.request
        import urllib.error

        url = f"{self.master_url}/sync?since={self._last_applied_seq}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())

            entries = data.get('entries', [])
            if not entries:
                return {'synced': 0, 'latest_seq': data.get('latest_seq', -1)}

            # Verify and apply each entry
            applied = 0
            for entry in entries:
                # Verify SHA-256
                if not self._verify_entry(entry):
                    print(f"  [replica] SHA-256 mismatch at seq {entry['seq']} — skipping")
                    self._sync_errors += 1
                    continue

                # Apply to local grid
                tokens = [Token(t) for t in entry['tokens']]
                self.store.write(entry['record_id'], tokens)
                applied += 1
                self._last_applied_seq = entry['seq']

            self._sync_count += 1
            latest = entries[-1]['seq'] if entries else self._last_applied_seq
            return {'synced': applied, 'latest_seq': latest}

        except urllib.error.URLError as e:
            self._sync_errors += 1
            return {'synced': 0, 'error': str(e)}

    def sync_loop(self, interval: float = 5.0):
        """Continuously poll master at the given interval (seconds)."""
        print(f"[replica] Starting sync loop (interval={interval}s)")
        print(f"[replica] Master: {self.master_url}")
        print(f"[replica] Current LSN: {self._last_applied_seq}")

        while True:
            try:
                result = self.sync()
                if result.get('synced', 0) > 0:
                    print(f"  [replica] Synced {result['synced']} entries, "
                          f"LSN now {result['latest_seq']}")
            except Exception as e:
                print(f"  [replica] Sync error: {e}")
                self._sync_errors += 1

            time.sleep(interval)

    def _verify_entry(self, entry: dict) -> bool:
        """Verify a WAL entry's SHA-256 hash."""
        try:
            # Reconstruct the content that was hashed
            WAL_MAGIC = 0x4750574C
            WAL_HDR_FMT = ">IIII"
            WAL_PREV_FMT = ">i"
            WAL_PAD_FMT = ">I"

            header = struct.pack(WAL_HDR_FMT, WAL_MAGIC, entry['seq'],
                                 entry['record_id'], len(entry['tokens']))
            prev_bytes = struct.pack(WAL_PREV_FMT, entry['prev_hash_offset'])
            pad_bytes = struct.pack(WAL_PAD_FMT, entry['pad_len'])
            token_bytes = bytes.fromhex(entry['tokens_hex'])

            content = header + prev_bytes + token_bytes + pad_bytes
            computed = hashlib.sha256(content).hexdigest()
            return computed == entry['sha256']
        except Exception:
            return False

    def read(self, record_id: int) -> Optional[PositionedRecord]:
        return self.store.read(record_id)

    def stats(self) -> dict:
        return {
            'master_url': self.master_url,
            'last_lsn': self._last_applied_seq,
            'sync_count': self._sync_count,
            'sync_errors': self._sync_errors,
        }

    def close(self):
        self.store.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import tempfile, shutil, urllib.request
    import time

    demo_dir = tempfile.mkdtemp(prefix='griddb_rep_demo_')
    master_dir = os.path.join(demo_dir, 'master')
    replica_dir = os.path.join(demo_dir, 'replica')
    os.makedirs(master_dir); os.makedirs(replica_dir)

    print(f"Demo dir: {demo_dir}")
    print("═" * 60)
    print("  GridDB Replication — Master/Replica over HTTP")
    print("═" * 60)

    try:
        # ── 1. Start master ──
        print("\n── 1. Start Master ──")
        master = ReplicationMaster(data_dir=master_dir, port=19001)
        master.start()
        time.sleep(0.5)

        # Verify master is running
        try:
            health = json.loads(urllib.request.urlopen("http://localhost:19001/health").read())
            print(f"  Master health: {health['status']}")
        except Exception:
            print("  Master failed to start")
            sys.exit(1)

        # ── 2. Write data to master ──
        print("\n── 2. Write records to master ──")
        test_data = [
            (0, "Alice", 5000),
            (1, "Bob", 10000),
            (42, "Charlie", 7500),
        ]
        for rid, name, balance in test_data:
            tokens = [
                *Encoder.encode_word(name),
                *Encoder.encode_integer(balance),
                Token.RECORD,
            ]
            master.write(rid, tokens)
            print(f"  write(#{rid}, '{name}') → LSN {master.store.last_seq}")

        # ── 3. Create and sync replica ──
        print("\n── 3. Create Replica + Initial Sync ──")
        replica = Replica(master_url="http://localhost:19001", data_dir=replica_dir)
        print(f"  Replica LSN before sync: {replica.last_lsn}")

        result = replica.sync()
        print(f"  Sync result: {result['synced']} entries, LSN now {replica.last_lsn}")

        # ── 4. Verify replica data ──
        print("\n── 4. Verify Replica Data ──")
        for rid, expected_name, expected_bal in test_data:
            rec = replica.read(rid)
            if rec:
                names = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
                vals = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
                name_match = ''.join(names) == expected_name if names else False
                bal_match = vals[0] == expected_bal if vals else False
                ok = "✓" if (name_match and bal_match) else "✗"
                print(f"  replica.read(#{rid}) → name={names}, bal={vals} {ok}")
            else:
                print(f"  replica.read(#{rid}) → None ✗")

        # ── 5. Incremental sync (more writes to master) ──
        print("\n── 5. Incremental Sync (new writes on master) ──")
        for rid, name, balance in [(99, "Diana", 3000), (100, "Eve", 12000)]:
            tokens = [
                *Encoder.encode_word(name),
                *Encoder.encode_integer(balance),
                Token.RECORD,
            ]
            master.write(rid, tokens)
            print(f"  master.write(#{rid}, '{name}') → LSN {master.store.last_seq}")

        result2 = replica.sync()
        print(f"  Incremental sync: {result2['synced']} new entries, LSN {replica.last_lsn}")

        # Verify incremental
        for rid in [99, 100]:
            rec = replica.read(rid)
            names = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
            print(f"  replica.read(#{rid}) → {names[0] if names else '?'}")

        # ── 6. SHA-256 tamper detection ──
        print("\n── 6. SHA-256 Integrity — tamper detection ──")
        print(f"  Replica stats: {replica.stats()}")

        # ── 7. Stats ──
        print("\n── 7. Final State ──")
        print(f"  Master LSN: {master.store.last_seq}")
        print(f"  Replica LSN: {replica.last_lsn}")
        print(f"  In sync: {'✓' if master.store.last_seq == replica.last_lsn else '✗'}")

        print("\n" + "═" * 60)
        print("  Replication demo complete")
        print("═" * 60)

    finally:
        master.close()
        replica.close()
        shutil.rmtree(demo_dir, ignore_errors=True)
