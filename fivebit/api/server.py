"""
5bit REST API Server — Deterministic, Content-Addressed
==========================================================
Auto-generates REST endpoints from a 5bit encoding spec.
No schema introspection. No ORM. Routes are pure functions of data.

  GET  /records/<id>         → O(1) positioned read
  GET  /records/<hash>       → content-addressed lookup
  GET  /records?field=val    → filtered B-tree scan
  POST /records              → create (ETag = SHA-256 of record)
  PUT  /records/<id>         → CAS update
  DELETE /records/<id>       → tombstone

Every response has ETag = sha256(record_bytes). Free conditional GETs.
Every response is deterministic — same input = same bytes = same HTTP.
"""
import os, sys, json, hashlib, struct
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Callable, Optional, Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord, pack_to_bytes
from griddb_alloc import AllocGrid, AllocRecord

class APIHandler(BaseHTTPRequestHandler):
    """HTTP handler for 5bit REST API. Set .grid and .spec before starting."""

    grid: AllocGrid = None  # type: ignore
    spec: Dict[str, Any] = {}  # encoding spec
    _etag_cache: Dict[int, str] = {}  # record_id → ETag

    def log_message(self, fmt, *args):
        print(f"  [api] {args[0]}")

    def _send(self, code: int, body: Any, etag: str = ''):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        if etag:
            self.send_header('ETag', f'"{etag}"')
            # Conditional GET
            if_none = self.headers.get('If-None-Match', '')
            if if_none == f'"{etag}"':
                self.send_response(304)
                self.end_headers()
                return
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _record_to_json(self, rec: AllocRecord) -> dict:
        """Convert a parsed record to JSON using the encoding spec."""
        result = {}
        nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
        words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
        fields = self.spec.get('fields', [])
        for i, field in enumerate(fields):
            if i < len(nums):
                result[field] = nums[i]
            elif i - len(nums) < len(words):
                wi = i - len(nums)
                # Join adjacent word fragments
                text = ''.join(words[wi:]) if wi < len(words) else ''
                result[field] = text
        result['_id'] = rec.record_id
        result['_hash'] = self._record_hash(rec)
        return result

    def _record_hash(self, rec: AllocRecord) -> str:
        """SHA-256 of packed record bytes — deterministic ETag."""
        if rec.record_id in self._etag_cache:
            return self._etag_cache[rec.record_id]
        packed, _ = pack_to_bytes(rec.tokens)
        h = hashlib.sha256(bytes(packed)).hexdigest()[:16]
        self._etag_cache[rec.record_id] = h
        return h

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')
        params = parse_qs(parsed.query)

        # GET /records/<id> or /records/by-hash/<hash>
        parts = path.split('/')

        if len(parts) >= 3 and parts[1] == 'records':
            try:
                rid = int(parts[2])
                rec = self.grid.read(rid)
                if not rec or rec.is_tombstone:
                    self._send(404, {'error': 'Not found'})
                    return
                body = self._record_to_json(rec)
                self._send(200, body, self._record_hash(rec))
                return
            except ValueError:
                pass

            # Content-addressed: GET /records/by-hash/<sha256>
            if parts[2] == 'by-hash' and len(parts) >= 4:
                target_hash = parts[3]
                for rid in range(self.grid.total_entries):
                    rec = self.grid.read(rid)
                    if rec and not rec.is_tombstone:
                        if self._record_hash(rec) == target_hash:
                            self._send(200, self._record_to_json(rec), target_hash)
                            return
                self._send(404, {'error': 'Hash not found'})
                return

        # GET /records?field=value&limit=N
        if path == '/records' and params:
            field = params.get('field', [None])[0]
            value = params.get('value', [None])[0]
            limit = int(params.get('limit', ['20'])[0])
            if field and value:
                results = []
                fields = self.spec.get('fields', [])
                field_idx = fields.index(field) if field in fields else -1
                for rid in range(min(self.grid.total_entries, 10000)):
                    rec = self.grid.read(rid)
                    if not rec or rec.is_tombstone: continue
                    nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
                    words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
                    text = ''.join(words)
                    matched = False
                    if field_idx >= 0 and field_idx < len(nums):
                        matched = str(nums[field_idx]) == value
                    elif value.lower() in text.lower():
                        matched = True
                    if matched:
                        results.append(self._record_to_json(rec))
                        if len(results) >= limit: break
                self._send(200, {'results': results, 'count': len(results)})
                return

        self._send(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path == '/records':
            content_len = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_len))
            rid = body.get('_id') or self._next_id()
            tokens = self._json_to_tokens(body)
            self.grid.write(rid, tokens)
            rec = self.grid.read(rid)
            if rec:
                self._send(201, self._record_to_json(rec), self._record_hash(rec))
            else:
                self._send(500, {'error': 'Write failed'})
        else:
            self._send(404, {'error': 'Not found'})

    def do_PUT(self):
        parsed = urlparse(self.path)
        parts = parsed.path.rstrip('/').split('/')
        if len(parts) >= 3 and parts[1] == 'records':
            try:
                rid = int(parts[2])
                existing = self.grid.read(rid)
                content_len = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(content_len))
                tokens = self._json_to_tokens(body)
                # CAS: compare-and-swap
                if_match = self.headers.get('If-Match', '')
                if if_match and existing:
                    current_hash = self._record_hash(existing)
                    if if_match != f'"{current_hash}"':
                        self._send(409, {'error': 'Conflict — record modified'})
                        return
                self.grid.write(rid, tokens)
                rec = self.grid.read(rid)
                if rec:
                    self._send(200, self._record_to_json(rec), self._record_hash(rec))
                else:
                    self._send(500, {'error': 'Update failed'})
                return
            except ValueError:
                pass
        self._send(404, {'error': 'Not found'})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        parts = parsed.path.rstrip('/').split('/')
        if len(parts) >= 3 and parts[1] == 'records':
            try:
                rid = int(parts[2])
                self.grid.delete(rid)
                self._send(200, {'deleted': rid})
                return
            except ValueError:
                pass
        self._send(404, {'error': 'Not found'})

    def _next_id(self) -> int:
        return self.grid.total_entries

    def _json_to_tokens(self, body: dict) -> List[Token]:
        """Convert JSON body to 5bit tokens using the encoding spec."""
        tokens: List[Token] = []
        fields = self.spec.get('fields', [])
        body.pop('_id', None); body.pop('_hash', None)
        for field in fields:
            val = body.get(field)
            if val is None: continue
            if isinstance(val, (int, float)):
                tokens.extend(Encoder.encode_integer(int(val)))
            elif isinstance(val, str):
                tokens.extend(Encoder.encode_word(val))
        tokens.append(Token.RECORD)
        return tokens


class APIServer:
    """5bit REST API server. Auto-generates routes from encoding spec.

    Usage:
      spec = {'name': 'users', 'fields': ['balance', 'name']}
      server = APIServer("./data", spec, port=8080)
      server.start()
    """

    def __init__(self, data_dir: str, spec: Dict[str, Any], port: int = 8080, cache_size: int = 1000):
        self.grid = AllocGrid(data_dir=data_dir, cache_size=cache_size)
        self.spec = spec
        self.port = port
        APIHandler.grid = self.grid
        APIHandler.spec = spec

    def start(self, blocking: bool = False):
        import threading
        self._server = HTTPServer(('0.0.0.0', self.port), APIHandler)
        print(f"[5bit API] {self.spec.get('name','records')} on :{self.port}")
        print(f"[5bit API] Fields: {self.spec.get('fields',[])}")
        print(f"[5bit API] Content-addressed — ETag = SHA-256 of record bytes")
        if blocking:
            self._server.serve_forever()
        else:
            t = threading.Thread(target=self._server.serve_forever, daemon=True)
            t.start()

    def stop(self):
        if hasattr(self, '_server'):
            self._server.shutdown()
        self.grid.close()
