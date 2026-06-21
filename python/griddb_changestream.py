#!/usr/bin/env python3
"""
GridDB Change Streams — Live Event Feed from the WAL
======================================================
Every write is a WAL entry.  Change streams tail the WAL
and emit structured events to subscribers.

  Subscriber: GET /stream?since=42
  Master:     event: {seq:43, type:"PUT", record_id:0, data:{name:"Alice", balance:5000}}
              event: {seq:44, type:"TXN_COMMIT", txn_id:1}
              ...

Same WAL that powers replication.  Different consumer.

Supports:
  - HTTP Server-Sent Events (SSE) — push to browsers
  - Long-poll — simple HTTP clients
  - Filtering by record_id, event type
  - Resume from any sequence number

Usage:
  # Start change stream server
  server = ChangeStreamServer(wal_path="./data/txn_wal.grid", port=9002)
  server.start()

  # Subscribe (SSE)
  curl -N http://localhost:9002/stream?since=0

  # Long-poll
  curl http://localhost:9002/poll?since=0
"""

import os
import sys
import json
import struct
import hashlib
import time
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Optional, Callable, Dict, Any
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_grid_db import Token, Encoder, unpack_from_bytes
from griddb_transactions import TxnWAL


# ═══════════════════════════════════════════════════════════════════════════════
# Change Stream Engine
# ═══════════════════════════════════════════════════════════════════════════════

class ChangeStream:
    """Tails a WAL file and emits parsed events.

    Reads WAL entries, converts them to structured JSON events.
    Supports filtering and resume from any sequence number.
    """

    def __init__(self, wal: TxnWAL):
        self.wal = wal
        self._subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()
        self._last_seq = -1
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, poll_interval: float = 0.5):
        """Start tailing the WAL in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._tail_loop,
                                        args=(poll_interval,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def subscribe(self) -> queue.Queue:
        """Create a new subscriber queue. Returns a queue that receives events."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def get_events_since(self, since_seq: int,
                         filter_rid: Optional[int] = None,
                         filter_types: Optional[List[str]] = None) -> List[dict]:
        """Get all events since a sequence number. Supports filtering."""
        entries = self.wal.read_all()
        events = []

        for i, entry in enumerate(entries):
            seq = i  # Use index as sequence number
            if seq <= since_seq:
                continue

            event = self._entry_to_event(entry, seq)

            # Apply filters
            if filter_rid and event.get('record_id') != filter_rid:
                continue
            if filter_types and event.get('type') not in filter_types:
                continue

            events.append(event)

        return events

    def _entry_to_event(self, entry: dict, seq: int) -> dict:
        """Convert a raw WAL entry to a structured event."""
        tokens = entry.get('tokens', [])
        flags = entry.get('flags', 0)

        # Determine event type
        if flags == TxnWAL.FLAG_COMMITTED:
            return {
                'seq': seq,
                'type': 'TXN_COMMIT',
                'txn_id': entry['txn_id'],
                'timestamp': int(time.time() * 1000),
            }
        elif flags == TxnWAL.FLAG_PENDING:
            # Parse tokens to extract data
            parsed = self._parse_tokens(tokens)
            record_id = entry.get('record_id', -1)
            return {
                'seq': seq,
                'type': parsed.get('op', 'PUT'),
                'txn_id': entry['txn_id'],
                'record_id': record_id,
                'data': parsed.get('data', {}),
                'tokens': [int(t) for t in tokens],
                'timestamp': int(time.time() * 1000),
            }

        return {
            'seq': seq,
            'type': 'UNKNOWN',
            'flags': flags,
            'timestamp': int(time.time() * 1000),
        }

    def _parse_tokens(self, tokens: List) -> dict:
        """Best-effort parse of tokens to extract data fields."""
        result: Dict[str, Any] = {'op': 'PUT', 'data': {}}
        nums = []
        words = []

        for t in tokens:
            if isinstance(t, int):
                try:
                    tok = Token(t)
                except ValueError:
                    nums.append(t)
                    continue
                from binary_grid_db import NUMERIC_DIGIT_VALUE, WORD_CHAR
                if tok in NUMERIC_DIGIT_VALUE and NUMERIC_DIGIT_VALUE[tok] is not None:
                    nums.append(NUMERIC_DIGIT_VALUE[tok])
                elif tok in WORD_CHAR:
                    words.append(WORD_CHAR[tok])
                elif tok == Token.END:
                    pass
                elif tok == Token.RECORD:
                    pass

        # Reconstruct numbers from signed digits
        if nums:
            value = 0
            n = len(nums)
            for i, d in enumerate(nums):
                value += d * (10 ** (n - 1 - i))
            result['data']['value'] = value

        if words:
            result['data']['text'] = ''.join(words)

        return result

    def _tail_loop(self, poll_interval: float):
        """Background loop: tail WAL, emit new events to subscribers."""
        while self._running:
            entries = self.wal.read_all()
            new_entries = entries[self._last_seq + 1:]

            for i, entry in enumerate(new_entries):
                seq = self._last_seq + 1 + i
                event = self._entry_to_event(entry, seq)
                self._broadcast(event)

            if new_entries:
                self._last_seq = len(entries) - 1

            time.sleep(poll_interval)

    def _broadcast(self, event: dict):
        """Send event to all subscribers."""
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Server (SSE + Long-Poll)
# ═══════════════════════════════════════════════════════════════════════════════

class StreamHandler(BaseHTTPRequestHandler):
    """HTTP handler serving change stream events via SSE or long-poll."""

    stream: ChangeStream = None  # type: ignore

    def log_message(self, format, *args):
        print(f"  [stream] {args[0]}")

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/stream':
            self._handle_sse(parsed)
        elif parsed.path == '/poll':
            self._handle_poll(parsed)
        elif parsed.path == '/health':
            self.send_response(200); self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404); self.end_headers()

    def _handle_sse(self, parsed):
        """Server-Sent Events — push events to browser."""
        params = parse_qs(parsed.query)
        since = int(params.get('since', ['-1'])[0])

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Send historical events first
        events = self.stream.get_events_since(since)
        for event in events:
            self._send_sse(event)

        # Subscribe to live events
        q = self.stream.subscribe()
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                    self._send_sse(event)
                    q.task_done()
                except queue.Empty:
                    # Send keepalive comment
                    self.wfile.write(b': keepalive\n\n')
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.stream.unsubscribe(q)

    def _handle_poll(self, parsed):
        """Long-poll — return events as JSON array."""
        params = parse_qs(parsed.query)
        since = int(params.get('since', ['-1'])[0])
        timeout = int(params.get('timeout', ['30'])[0])

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Wait for new events or timeout
        start = time.time()
        q = self.stream.subscribe()
        events = []
        try:
            while time.time() - start < timeout:
                try:
                    event = q.get(timeout=1)
                    events.append(event)
                    q.task_done()
                    if events:
                        break
                except queue.Empty:
                    pass
        finally:
            self.stream.unsubscribe(q)

        # If no live events, return historical
        if not events:
            events = self.stream.get_events_since(since)

        self.wfile.write(json.dumps({
            'events': events,
            'count': len(events),
        }).encode())

    def _send_sse(self, event: dict):
        """Send one SSE event."""
        data = json.dumps(event)
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()


class ChangeStreamServer:
    """HTTP server that serves change stream events."""

    def __init__(self, wal: TxnWAL, port: int = 9002):
        self.wal = wal
        self.port = port
        self.stream = ChangeStream(wal)
        StreamHandler.stream = self.stream
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = False):
        self.stream.start()
        self._server = HTTPServer(('0.0.0.0', self.port), StreamHandler)
        print(f"[changestream] Listening on :{self.port}")
        print(f"[changestream] Tailing WAL: {self.wal.path}")

        if blocking:
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()

    def stop(self):
        self.stream.stop()
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import tempfile, shutil, urllib.request
    import time

    demo_dir = tempfile.mkdtemp(prefix='griddb_stream_demo_')
    print(f"Demo dir: {demo_dir}")
    print("═" * 60)
    print("  GridDB Change Streams — Live WAL Events")
    print("═" * 60)

    try:
        from griddb_transactions import TransactionalGrid

        # ── 1. Create grid and generate WAL entries ──
        print("\n── 1. Generate WAL events ──")
        tgrid = TransactionalGrid(data_dir=demo_dir)

        # Autocommit writes (no transaction)
        tgrid.put(0, [
            *Encoder.encode_word("Alice"),
            *Encoder.encode_integer(10000),
            Token.RECORD,
        ])
        print("  Event: PUT Alice, $100")

        # Transactional write
        txn = tgrid.begin()
        txn.put(1, [
            *Encoder.encode_word("Bob"),
            *Encoder.encode_integer(5000),
            Token.RECORD,
        ])
        tgrid.commit()
        print("  Event: PUT Bob, $50 (in transaction)")

        # Another transaction
        txn2 = tgrid.begin()
        txn2.put(0, [
            *Encoder.encode_word("Alice"),
            *Encoder.encode_integer(3000),
            Token.RECORD,
        ])
        txn2.put(2, [
            *Encoder.encode_word("Carol"),
            *Encoder.encode_integer(7000),
            Token.RECORD,
        ])
        tgrid.commit()
        print("  Event: PUT Alice→$30, PUT Carol $70 (multi-write txn)")

        # ── 2. Start change stream server ──
        print("\n── 2. Start Change Stream Server ──")
        server = ChangeStreamServer(wal=tgrid.wal, port=19002)
        server.start()
        time.sleep(0.5)

        # ── 3. Poll for historical events ──
        print("\n── 3. Poll: Get all events since seq 0 ──")
        resp = urllib.request.urlopen("http://localhost:19002/poll?since=-1")
        data = json.loads(resp.read())
        for event in data['events']:
            print(f"  seq={event['seq']}: {event['type']} "
                  f"rid={event.get('record_id','-')} "
                  f"txn={event.get('txn_id','-')} "
                  f"data={event.get('data',{})}")

        # ── 4. Live SSE stream (capture 1 event) ──
        print("\n── 4. SSE: Live event after new write ──")

        # Write new data
        tgrid.put(3, [
            *Encoder.encode_word("Diana"),
            *Encoder.encode_integer(9000),
            Token.RECORD,
        ])
        print("  Wrote: PUT Diana $90 (outside txn)")

        time.sleep(1)  # Let the tail loop pick it up

        # Long-poll to catch the new event
        resp = urllib.request.urlopen(
            f"http://localhost:19002/poll?since={data['events'][-1]['seq']}&timeout=5")
        new_data = json.loads(resp.read())
        print(f"  Poll caught {new_data['count']} new event(s):")
        for event in new_data['events']:
            print(f"  seq={event['seq']}: {event['type']} "
                  f"rid={event.get('record_id','-')} "
                  f"data={event.get('data',{})}")

        # ── 5. Filtered poll ──
        print(f"\n── 5. Filtered: Events for record_id=0 only ──")
        filtered = server.stream.get_events_since(-1, filter_rid=0)
        for event in filtered:
            print(f"  seq={event['seq']}: rid={event.get('record_id')} "
                  f"data={event.get('data',{})}")

        print(f"\n── Summary ──")
        print(f"  Total events in WAL: {len(server.stream.get_events_since(-1))}")
        print(f"  Filtered (rid=0): {len(filtered)}")
        print(f"  Change stream = WAL exposed to application")

        tgrid.close()
        server.stop()

        print("\n" + "═" * 60)
        print("  Change Streams demo complete")
        print("═" * 60)

    finally:
        shutil.rmtree(demo_dir, ignore_errors=True)
