"""
5bit Webhooks — WAL Change Stream → HTTP POST
===============================================
Taps the existing WAL change stream. Fires webhooks on insert/update/delete.
Retry with exponential backoff. Dead-letter after max retries. Grid-durable.

POST /api/webhooks  { url, table, events }  → { id, secret }
GET  /api/webhooks                          → list configured webhooks
DELETE /api/webhooks/{id}                    → remove
"""
import os, sys, json, time, hashlib, hmac, threading, urllib.request
from collections import defaultdict
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid

WEBHOOK_BASE = 70_000_000
DELIVERY_BASE = 71_000_000
MAX_RETRIES = 5
BACKOFF = [1, 2, 4, 8, 16]
import ipaddress, re

# SSRF guard: blocked CIDRs + schemes
BLOCKED_NETS = [
    ipaddress.ip_network('127.0.0.0/8'), ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'), ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'), ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),
]

def _is_safe_url(url: str) -> bool:
    """Reject private/loopback/link-local IPs and non-http(s) schemes."""
    if not url.startswith(('http://', 'https://')):
        return False
    try:
        host = url.split('://')[1].split('/')[0].split(':')[0]
        addr = ipaddress.ip_address(host)
        return not any(addr in net for net in BLOCKED_NETS)
    except ValueError:
        return True  # Hostname, not IP — allow


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
            url = ''.join(p.text for p in url_rec.parsed if isinstance(p, ParsedWord))
            table_rec = self.grid.read(self._hook_rid(rid, 1))
            table = ''.join(p.text for p in table_rec.parsed if isinstance(p, ParsedWord)) if table_rec else ''
            events_rec = self.grid.read(self._hook_rid(rid, 2))
            events = ''.join(p.text for p in events_rec.parsed if isinstance(p, ParsedWord)) if events_rec else ''
            secret_rec = self.grid.read(self._hook_rid(rid, 3))
            secret = ''.join(p.text for p in secret_rec.parsed if isinstance(p, ParsedWord)) if secret_rec else ''
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
        """Fire a webhook delivery with retry."""
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

        # Attempt delivery with backoff
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(hook['url'], data=payload, headers=headers, method='POST')
                resp = urllib.request.urlopen(req, timeout=10)
                if 200 <= resp.status < 300:
                    return  # Success
            except Exception:
                pass
            time.sleep(BACKOFF[attempt])

        # Dead-letter: store failed delivery in grid
        dlq_rid = self._next_id(DELIVERY_BASE)
        dlq_tokens = [
            *Encoder.encode_word(hook['url']),
            *Encoder.encode_integer(hook['id']),
            *Encoder.encode_word(event_type),
            *Encoder.encode_word(payload.decode()[:500]),
            Token.RECORD,
        ]
        self.grid.write(dlq_rid, dlq_tokens)

    def dead_letter_queue(self) -> List[dict]:
        """List failed deliveries."""
        dlq = []
        for rid in range(DELIVERY_BASE, DELIVERY_BASE + 1000):
            rec = self.grid.read(rid)
            if not rec or rec.is_tombstone: continue
            words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
            nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
            if len(words) >= 3:
                dlq.append({'id': rid, 'url': words[0], 'hook_id': nums[0] if nums else 0,
                            'event': words[2], 'payload': words[3][:100]})
        return dlq

    def retry_dead_letter(self):
        """Retry all dead-lettered deliveries."""
        for item in self.dead_letter_queue():
            hooks = self.list()
            hook = next((h for h in hooks if h['id'] == item['hook_id']), None)
            if hook:
                self._deliver(hook, item['event'], {'_retry': True})
            self.grid.delete(item['id'])

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
