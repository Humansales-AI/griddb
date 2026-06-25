"""
5bit Storage — Content-Addressed File Store
=============================================
Files stored as records in the grid. SHA-256 = dedup key.
RLS-gated. Append-only with tombstone deletes.

POST   /storage/{bucket}           multipart → { path, hash, size }
GET    /storage/{bucket}/{path}    download (ETag = SHA-256)
DELETE /storage/{bucket}/{path}    tombstone
GET    /storage/{bucket}?prefix=   list objects
"""
import os, sys, json, hashlib, io
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid, AllocRecord

BUCKET_STRIDE = 10_000_000  # bucket_name hash → record space

def _bucket_base(name: str) -> int:
    h = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(h[:4], 'big') % (BUCKET_STRIDE // 2)

class StorageServer:
    """Content-addressed file storage on 5bit grid."""

    def __init__(self, data_dir: str = "./data", cache_size: int = 1000):
        self.grid = AllocGrid(data_dir=data_dir, cache_size=cache_size)
        # File index: bucket_base + path_hash → (owner, size, content_sha256, record_ids)
        self._owner_id: Optional[int] = None  # Set by auth middleware

    def set_owner(self, uid: int): self._owner_id = uid

    def _path_rid(self, bucket: str, path: str) -> int:
        base = _bucket_base(bucket)
        h = hashlib.sha256(f"{bucket}/{path}".encode()).digest()
        return base * BUCKET_STRIDE + (int.from_bytes(h[:4], 'big') % (BUCKET_STRIDE - 1))

    def upload(self, bucket: str, path: str, data: bytes, content_type: str = '') -> dict:
        """Upload a file. Returns { path, hash, size }. Deduplicates by SHA-256."""
        content_hash = hashlib.sha256(data).hexdigest()
        rid = self._path_rid(bucket, path)

        # Check existing
        existing = self.grid.read(rid)
        if existing and not existing.is_tombstone:
            existing_hash = self._extract_hash(existing)
            if existing_hash == content_hash:
                return {'path': path, 'hash': content_hash, 'size': len(data), 'dedup': True}

        # Store metadata record
        meta_tokens = [
            *Encoder.encode_word(path),
            *Encoder.encode_word(bucket),
            *Encoder.encode_integer(len(data)),
            *Encoder.encode_word(content_hash),
            *Encoder.encode_integer(self._owner_id or 0),
            Token.RECORD,
        ]
        self.grid.write(rid, meta_tokens)

        # Store content in chunks (max ~200 tokens per record to fit stride)
        CHUNK = 150
        content_tokens: List[Token] = []
        for b in data:
            content_tokens.extend(Encoder.encode_integer(b))
        chunks = [content_tokens[i:i+CHUNK] for i in range(0, len(content_tokens), CHUNK)]
        for ci, chunk in enumerate(chunks):
            chunk.append(Token.RECORD)
            self.grid.write(rid + 1 + ci, chunk)

        return {'path': path, 'hash': content_hash, 'size': len(data), 'chunks': len(chunks)}

    def download(self, bucket: str, path: str) -> Optional[bytes]:
        """Download a file. Returns raw bytes."""
        rid = self._path_rid(bucket, path)
        meta = self.grid.read(rid)
        if not meta or meta.is_tombstone:
            return None

        # Read content chunks
        data = bytearray()
        ci = 1
        while True:
            chunk_rec = self.grid.read(rid + ci)
            if not chunk_rec or chunk_rec.is_tombstone:
                break
            nums = [p.value for p in chunk_rec.parsed if isinstance(p, ParsedNumber)]
            data.extend(bytes(nums))
            ci += 1

        return bytes(data)

    def delete(self, bucket: str, path: str) -> bool:
        """Tombstone a file."""
        rid = self._path_rid(bucket, path)
        meta = self.grid.read(rid)
        if not meta: return False
        self.grid.delete(rid)
        ci = 1
        while True:
            chunk = self.grid.read(rid + ci)
            if not chunk or chunk.is_tombstone: break
            self.grid.delete(rid + ci)
            ci += 1
        return True

    def list_objects(self, bucket: str, prefix: str = '', limit: int = 100) -> List[dict]:
        """List files in a bucket, optionally filtered by prefix."""
        results = []
        base = _bucket_base(bucket) * BUCKET_STRIDE
        for rid in range(base, base + BUCKET_STRIDE):
            rec = self.grid.read(rid)
            if not rec or rec.is_tombstone: continue
            words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
            nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
            if len(words) >= 2 and words[1] == bucket:
                file_path = words[0]
                if prefix and not file_path.startswith(prefix): continue
                results.append({
                    'path': file_path,
                    'size': nums[0] if nums else 0,
                    'hash': words[3] if len(words) >= 4 else '',
                    'owner': nums[1] if len(nums) >= 2 else 0,
                })
                if len(results) >= limit: break
        return results

    def close(self): self.grid.close()

    def _extract_hash(self, rec: AllocRecord) -> str:
        words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
        return words[3] if len(words) >= 4 else ''
