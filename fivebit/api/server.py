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
    rate_limiter = None  # Set by APIServer

    def log_message(self, fmt, *args):
        print(f"  [api] {args[0]}")

    def _check_rate(self) -> bool:
        """Rate limit check. Returns True if allowed."""
        if self.rate_limiter is None:
            return True
        ip = self.client_address[0]
        allowed, retry = self.rate_limiter.check(ip)
        if not allowed:
            self.send_response(429)
            self.send_header('Retry-After', str(retry))
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Rate limit exceeded', 'retryAfter': retry}).encode())
        return allowed

    def _send(self, code: int, body: Any, etag: str = ''):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        # Check conditional GET BEFORE writing status
        if etag:
            if_none = self.headers.get('If-None-Match', '').strip('"')
            if if_none == etag:
                self.send_response(304)
                self.end_headers()
                return
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        if etag:
            self.send_header('ETag', f'"{etag}"')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _record_to_json(self, rec: AllocRecord) -> dict:
        """Convert a parsed record to JSON. Pairs values to spec fields in parse order."""
        result = {}
        # Walk parsed output in order, collecting nums + words as they appear
        values: List[Any] = []
        pending_text = ''
        for p in rec.parsed:
            if isinstance(p, ParsedNumber):
                if pending_text:
                    values.append(pending_text)
                    pending_text = ''
                values.append(p.value)
            elif isinstance(p, ParsedWord):
                pending_text += p.text
        if pending_text:
            values.append(pending_text)

        fields = self.spec.get('fields', [])
        for i, field in enumerate(fields):
            if i < len(values):
                result[field] = values[i]
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
        if not self._check_rate(): return
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

        # GET /query?filter=age:gt:21&aggregate=count
        if path == '/query':
            self._handle_query(params)
            return

        self._send(404, {'error': 'Not found'})

    def _handle_query(self, params: dict):
        """Single-pass scan with filters + aggregates."""
        filters = self._parse_filters(params.get('filter', []))
        aggs = params.get('aggregate', [])

        # Collect matching records
        results = []
        fields = self.spec.get('fields', [])
        for rid in range(min(self.grid.total_entries, 100000)):
            rec = self.grid.read(rid)
            if not rec or rec.is_tombstone: continue
            record = self._record_to_dict(rec, fields)
            if self._matches_filters(record, filters):
                results.append(record)

        # Compute aggregates
        output = {'count': len(results)}
        for agg in aggs:
            parts = agg.split(':')
            op = parts[0]  # count, sum, avg, min, max
            field = parts[1] if len(parts) > 1 else None
            vals = [r[field] for r in results if field and field in r and isinstance(r[field], (int, float))]
            if op == 'count':
                output['count'] = len(results)
            elif op == 'sum' and vals:
                output[f'sum_{field}'] = sum(vals)
            elif op == 'avg' and vals:
                output[f'avg_{field}'] = sum(vals) / len(vals)
            elif op == 'min' and vals:
                output[f'min_{field}'] = min(vals)
            elif op == 'max' and vals:
                output[f'max_{field}'] = max(vals)

        self._send(200, output)

    def _parse_filters(self, filter_list: list) -> list:
        """Parse filter strings like 'age:gt:21' into structured filters."""
        filters = []
        for f in filter_list:
            parts = f.split(':')
            if len(parts) == 3:
                filters.append({'field': parts[0], 'op': parts[1], 'value': parts[2]})
            elif len(parts) == 2:
                filters.append({'field': parts[0], 'op': 'eq', 'value': parts[1]})
        return filters

    def _matches_filters(self, record: dict, filters: list) -> bool:
        for f in filters:
            val = record.get(f['field'])
            if val is None: return False
            target = f['value']
            op = f['op']
            try:
                if isinstance(val, (int, float)):
                    target_num = float(target)
                    if op == 'eq': return val == target_num
                    if op == 'gt': return val > target_num
                    if op == 'gte': return val >= target_num
                    if op == 'lt': return val < target_num
                    if op == 'lte': return val <= target_num
                else:
                    val_str = str(val).lower()
                    t_str = target.lower()
                    if op == 'eq': return val_str == t_str
                    if op == 'contains': return t_str in val_str
                    if op == 'startsWith': return val_str.startswith(t_str)
            except ValueError:
                return False
            return True
        return True

    def _record_to_dict(self, rec, fields: list) -> dict:
        """Convert record to dict using spec fields."""
        result = {}
        vals = []
        pending = ''
        for p in rec.parsed:
            if isinstance(p, ParsedNumber):
                if pending: vals.append(pending); pending = ''
                vals.append(p.value)
            elif isinstance(p, ParsedWord):
                pending += p.text
        if pending: vals.append(pending)
        for i, field in enumerate(fields):
            if i < len(vals): result[field] = vals[i]
        result['_id'] = rec.record_id
        return result

    def do_POST(self):
        if not self._check_rate(): return
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
        if not self._check_rate(): return
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
                        self._send(412, {'error': 'Precondition Failed — record modified'})
                        return
                self._etag_cache.pop(rid, None)  # invalidate stale ETag
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
        if not self._check_rate(): return
        parsed = urlparse(self.path)
        parts = parsed.path.rstrip('/').split('/')
        if len(parts) >= 3 and parts[1] == 'records':
            try:
                rid = int(parts[2])
                self._etag_cache.pop(rid, None)
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
        from fivebit.api.ratelimit import APIRateLimiter
        APIHandler.rate_limiter = APIRateLimiter()

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
