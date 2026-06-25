"""
5bit Webhooks — WAL Change Stream → HTTP POST
===============================================
Taps the existing WAL change stream. Fires webhooks on insert/update/delete.
Retry with exponential backoff. Dead-letter after max retries. Grid-durable.

POST /api/webhooks  { url, table, events }  → { id, secret }
GET  /api/webhooks                          → list configured webhooks
DELETE /api/webhooks/{id}                    → remove
"""
import os, sys, json, time, hashlib, hmac, threading, urllib.request, socket
from collections import defaultdict
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid, AllocRecord

WEBHOOK_BASE = 70_000_000
DELIVERY_BASE = 71_000_000
MAX_RETRIES = 5
BACKOFF = [1, 2, 4, 8, 16]
import ipaddress, re

# SSRF guard: blocked CIDRs + resolve hostnames
BLOCKED_NETS = [
    ipaddress.ip_network('127.0.0.0/8'), ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'), ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'), ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),
]

def _is_safe_url(url: str) -> bool:
    """Reject private/loopback/link-local targets. Resolves hostnames."""
    if not url.startswith(('http://', 'https://')):
        return False
    try:
        host = url.split('://')[1].split('/')[0].split(':')[0]
        port = int(url.split(':')[-1].split('/')[0]) if ']:' not in url and url.count(':') >= 3 else None
        host = host.strip('[]')  # Strip IPv6 brackets
        # Resolve and check every address
        for info in socket.getaddrinfo(host, port or 80):
            addr = ipaddress.ip_address(info[4][0])
            if any(addr in net for net in BLOCKED_NETS):
                return False
        return True
    except Exception:
        return False  # Can't resolve = don't trust

def _reconstruct_all(rec: AllocRecord) -> str:
    """Reconstruct a string from ALL parsed tokens — words + numbers + specials."""
    result = ''
    for p in rec.parsed:
        if isinstance(p, ParsedWord):
            result += p.text
        elif isinstance(p, ParsedNumber):
            result += str(p.value)
    return result


class WebhookManager:
    """WAL-tail → HTTP POST. Config + deliveries stored in grid."""

    def __init__(self, grid: AllocGrid):
        self.grid = grid
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── CRUD (separate records per field — avoids word fragmentation) ─

    def _hook_rid(self, hook_id: int, field: int) -> int:
        return WEBHOOK_BASE + hook_id * 10 + field  # 0=url, 1=table, 2=events, 3=secret

    def create(self, url: str, table: str, events: List[str]) -> dict:
        """Register a webhook. Each field stored as separate record."""
        if not _is_safe_url(url):
            raise ValueError(f"Unsafe URL: {url} — private/loopback IPs blocked")
        rid = self._next_hook_id()
        secret = hashlib.sha256(os.urandom(32)).hexdigest()[:32]
        for fid, val in enumerate([url, table, ','.join(events), secret]):
            self.grid.write(self._hook_rid(rid, fid), [
                *Encoder.encode_word(val), Token.RECORD,
            ])
        return {'id': rid, 'secret': secret, 'url': url, 'table': table, 'events': events}

    def list(self) -> List[dict]:
        """List all configured webhooks."""
        hooks = []
        for rid in range(self._next_hook_id()):
            url_rec = self.grid.read(self._hook_rid(rid, 0))
            if not url_rec or url_rec.is_tombstone: continue
            url = _reconstruct_all(url_rec)
            table_rec = self.grid.read(self._hook_rid(rid, 1))
            table = _reconstruct_all(table_rec) if table_rec else ''
            events_rec = self.grid.read(self._hook_rid(rid, 2))
            events = _reconstruct_all(events_rec) if events_rec else ''
            secret_rec = self.grid.read(self._hook_rid(rid, 3))
            secret = _reconstruct_all(secret_rec) if secret_rec else ''
            hooks.append({'id': rid, 'url': url, 'table': table, 'events': events.split(','), 'secret': secret})
        return hooks

    def delete(self, hook_id: int) -> bool:
        for fid in range(4):
            rec = self.grid.read(self._hook_rid(hook_id, fid))
            if not rec: return False
            self.grid.delete(self._hook_rid(hook_id, fid))
        return True

    def _next_hook_id(self) -> int:
        rid = 0
        while self.grid.read(self._hook_rid(rid, 0)): rid += 1
        return rid

    # ── Delivery ───────────────────────────────────────────────────────

    def on_change(self, table: str, event_type: str, record: dict):
        """Called by WAL tail when a record changes."""
        hooks = self.list()
        for hook in hooks:
            if hook['table'] != table: continue
            if event_type not in hook['events']: continue
            self._deliver(hook, event_type, record)

    def _deliver(self, hook: dict, event_type: str, record: dict):
        """Fire a webhook delivery with retry. Re-validates URL at delivery time."""
        # Re-validate at delivery — prevents DNS rebinding (domain flips to private IP)
        if not _is_safe_url(hook['url']):
            return  # Silently drop — URL became unsafe since registration

        payload = json.dumps({
            'table': hook['table'],
            'event': event_type,
            'record': record,
            'timestamp': int(time.time()),
        }).encode()

        signature = hmac.new(
            hook['secret'].encode(), payload, 'sha256'
        ).hexdigest()

        headers = {
            'Content-Type': 'application/json',
            'X-Fivebit-Signature': f'sha256={signature}',
            'X-Fivebit-Event': event_type,
            'X-Fivebit-Table': hook['table'],
        }

        # Resolve once, pin IP, connect directly (prevents DNS rebinding at connect time)
        host = hook['url'].split('://')[1].split('/')[0]
        hostname = host.split(':')[0]
        port = int(host.split(':')[1]) if ':' in host else (443 if 'https' in hook['url'] else 80)
        scheme = 'https' if 'https' in hook['url'] else 'http'

        import http.client, ssl
        for attempt in range(MAX_RETRIES):
            try:
                # Resolve once, check, connect to pinned IP (Host header + SNI for name)
                addr = socket.getaddrinfo(hostname, port)[0][4][0]
                if any(ipaddress.ip_address(addr) in net for net in BLOCKED_NETS):
                    return  # Blocked at delivery
                headers['Host'] = hostname
                if scheme == 'https':
                    ctx = ssl.create_default_context()
                    sock = socket.create_connection((addr, port), timeout=10)
                    tls_sock = ctx.wrap_socket(sock, server_hostname=hostname)  # SNI + cert match
                    conn = http.client.HTTPSConnection(addr, port, context=ctx, timeout=10)
                    conn.sock = tls_sock
                else:
                    conn = http.client.HTTPConnection(addr, port, timeout=10)
                path = '/' + '/'.join(hook['url'].split('/', 3)[1:])
                conn.request('POST', path, body=payload, headers=headers)
                resp = conn.getresponse()
                if 200 <= resp.status < 300:
                    conn.close(); return
                conn.close()
            except Exception:
                pass
            time.sleep(BACKOFF[attempt])

        # Dead-letter: store failed delivery as separate fields (prevents fragmentation)
        dlq_rid = 0
        while self.grid.read(DELIVERY_BASE + dlq_rid * 10): dlq_rid += 1
        base = DELIVERY_BASE + dlq_rid * 10
        for fid, val in enumerate([hook['url'], str(hook['id']), event_type, payload.decode()[:500]]):
            self.grid.write(base + fid, [*Encoder.encode_word(val), Token.RECORD])

    def dead_letter_queue(self) -> List[dict]:
        """List failed deliveries."""
        dlq = []
        rid = 0
        while True:
            base = DELIVERY_BASE + rid * 10
            rec = self.grid.read(base)
            if not rec: break
            if not rec.is_tombstone:
                url = _reconstruct_all(rec)
                hook_rec = self.grid.read(base + 1)
                event_rec = self.grid.read(base + 2)
                payload_rec = self.grid.read(base + 3)
                dlq.append({
                    'id': rid,
                    'url': url,
                    'hook_id': int(_reconstruct_all(hook_rec)) if hook_rec else 0,
                    'event': _reconstruct_all(event_rec) if event_rec else '',
                    'payload': _reconstruct_all(payload_rec)[:100] if payload_rec else '',
                })
            rid += 1
        return dlq

    def retry_dead_letter(self):
        """Retry all dead-lettered deliveries."""
        for item in self.dead_letter_queue():
            hooks = self.list()
            hook = next((h for h in hooks if h['id'] == item['hook_id']), None)
            if hook:
                self._deliver(hook, item['event'], {'_retry': True})
            # Delete all 4 field records
            base = DELIVERY_BASE + item['id'] * 10
            for fid in range(4):
                self.grid.delete(base + fid)

    def start(self):
        """Background retry loop for dead-letter queue."""
        self._running = True
        def _loop():
            while self._running:
                time.sleep(30)
                self.retry_dead_letter()
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
