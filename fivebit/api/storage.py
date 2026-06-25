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

BUCKET_STRIDE = 10_000_000
MAX_RID = 2_000_000_000  # Keep within alloc table range

def _bucket_base(name: str) -> int:
    h = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(h[:4], 'big') % 1000  # Small range to avoid overflow

class StorageServer:
    """Content-addressed file storage on 5bit grid."""

    def __init__(self, data_dir: str = "./data", cache_size: int = 1000):
        self.grid = AllocGrid(data_dir=data_dir, cache_size=cache_size)
        # File index: bucket_base + path_hash → (owner, size, content_sha256, record_ids)
        self._owner_id: Optional[int] = None  # Set by auth middleware

    def set_owner(self, uid: int): self._owner_id = uid

    def _path_rid(self, bucket: str, path: str) -> int:
        h = hashlib.sha256(f"{bucket}/{path}".encode()).digest()
        return 10_000_000 + (int.from_bytes(h[:4], 'big') % 1_000_000)

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

    def _reconstruct(self, rec: AllocRecord) -> str:
        """Reconstruct a string from all parsed tokens (words + numbers)."""
        result = ''
        for p in rec.parsed:
            if isinstance(p, ParsedWord): result += p.text
            elif isinstance(p, ParsedNumber): result += str(p.value)
        return result

    OWNER_OFFSET = 1_000_000  # Separate record for owner to avoid field fragmentation

    def _get_owner(self, rid: int) -> int:
        """Read owner_id from SEPARATE record — immune to field fragmentation."""
        owner_rec = self.grid.read(rid + self.OWNER_OFFSET)
        if not owner_rec or owner_rec.is_tombstone: return -1
        nums = [p.value for p in owner_rec.parsed if isinstance(p, ParsedNumber)]
        return nums[0] if nums else -1

    def _set_owner(self, rid: int, owner_id: int):
        """Write owner_id to separate record."""
        from binary_grid_db import Encoder, Token
        self.grid.write(rid + self.OWNER_OFFSET, [
            *Encoder.encode_integer(owner_id), Token.RECORD,
        ])

    def upload(self, bucket: str, path: str, data: bytes, content_type: str = '') -> dict:
        """Upload a file. Returns { path, hash, size }. Deduplicates by SHA-256."""
        content_hash = hashlib.sha256(data).hexdigest()
        rid = self._path_rid(bucket, path)

        existing = self.grid.read(rid)
        if existing and not existing.is_tombstone:
            existing_hash = self._reconstruct(existing)
            if content_hash in existing_hash:
                return {'path': path, 'hash': content_hash, 'size': len(data), 'dedup': True}

        meta_tokens = [
            *Encoder.encode_word(path),
            *Encoder.encode_word(bucket),
            *Encoder.encode_integer(len(data)),
            *Encoder.encode_word(content_hash),
            Token.RECORD,
        ]
        self.grid.write(rid, meta_tokens)
        # Store owner in SEPARATE record
        if self._owner_id is not None:
            self._set_owner(rid, self._owner_id)

        content_tokens: List[Token] = []
        for b in data:
            content_tokens.extend(Encoder.encode_integer(b))
        chunks = [content_tokens[i:i+150] for i in range(0, len(content_tokens), 150)]
        for ci, chunk in enumerate(chunks):
            chunk.append(Token.RECORD)
            self.grid.write(rid + 1 + ci, chunk)

        return {'path': path, 'hash': content_hash, 'size': len(data), 'chunks': len(chunks)}

    def download(self, bucket: str, path: str) -> Optional[bytes]:
        """Download a file. Owner check enforced."""
        rid = self._path_rid(bucket, path)
        meta = self.grid.read(rid)
        if not meta or meta.is_tombstone:
            return None
        # Owner check — FAIL CLOSED
        stored_owner = self._get_owner(rid)
        if stored_owner < 0: return None  # No owner set = reject
        if self._owner_id is None or stored_owner != self._owner_id:
            return None

        data = bytearray()
        ci = 1
        while True:
            chunk_rec = self.grid.read(rid + ci)
            if not chunk_rec or chunk_rec.is_tombstone: break
            nums = [p.value for p in chunk_rec.parsed if isinstance(p, ParsedNumber)]
            data.extend(bytes(nums))
            ci += 1
        return bytes(data)

    def delete(self, bucket: str, path: str) -> bool:
        """Tombstone a file. Requires owner match."""
        rid = self._path_rid(bucket, path)
        meta = self.grid.read(rid)
        if not meta: return False
        stored = self._get_owner(rid)
        if stored < 0: return False  # No owner = reject
        if self._owner_id is None or stored != self._owner_id: return False
        self.grid.delete(rid)
        ci = 1
        while True:
            chunk = self.grid.read(rid + ci)
            if not chunk or chunk.is_tombstone: break
            self.grid.delete(rid + ci)
            ci += 1
        return True

    def list_objects(self, bucket: str, prefix: str = '', limit: int = 100) -> List[dict]:
        """List files in bucket. Shows only caller's files."""
        results = []
        # Scan a reasonable range
        for rid in range(10_000_000, 12_000_000):
            rec = self.grid.read(rid)
            if not rec or rec.is_tombstone: continue
            owner = self._get_owner(rid)
            if owner < 0: continue  # No owner set = skip
            if self._owner_id is None or owner != self._owner_id: continue
            # Check bucket name in text
            if bucket in text:
                path_str = text.split(bucket)[-1].lstrip('/')
                if prefix and not path_str.startswith(prefix): continue
                results.append({'path': path_str, 'size': nums[-1] if nums else 0,
                    'hash': hashlib.sha256(str(nums).encode()).hexdigest()[:16], 'owner': owner})
                if len(results) >= limit: break
        return results

    def close(self): self.grid.close()

    def _extract_hash(self, rec: AllocRecord) -> str:
        words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
        return words[2] if len(words) >= 3 else ''
