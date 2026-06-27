/**
 * AllocGrid — Two-Level O(1) Storage
 * ==================================
 * Level 1: alloc.grid (16 bytes/entry → record_id → offset+length+flags)
 * Level 2: data.grid (variable-length token blobs)
 *
 * read(42):  alloc[42] → (offset, length) → data.seek(offset) → O(1)
 * write(42): pack tokens → append to data → update alloc[42] → O(1)
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { Token, ParsedToken, ParsedNumber, ParsedWord, GridRecord } from './types';
import { Parser } from './parser';
import { packToBytes, unpackFromBytes } from './serialization';

const ALLOC_ENTRY_SIZE = 16;              // bytes per alloc entry
const ALLOC_MAGIC = 0x414C4F43;          // "ALOC"
const DATA_MAGIC = 0x44415441;           // "DATA"
const FLAG_FREE = 0;
const FLAG_ALLOCATED = 1;
const FLAG_TOMBSTONE = 2;

export interface AllocEntry {
  recordId: number;
  byteOffset: number;
  bitLength: number;
  flags: number;
}

export interface AllocRecord {
  recordId: number;
  tokens: Token[];
  parsed: ParsedToken[];
  byteOffset: number;
  bitLength: number;
  isTombstone: boolean;
}

export class AllocGrid {
  private allocPath: string;
  private dataPath: string;
  private dataEnd: number = 12; // header
  private allocFd: number | null = null;
  private dataFd: number | null = null;

  constructor(dataDir: string) {
    if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
    this.allocPath = path.join(dataDir, 'alloc.grid');
    this.dataPath = path.join(dataDir, 'data.grid');
    this._bootstrap();
  }

  private _bootstrap(): void {
    if (fs.existsSync(this.allocPath)) {
      const hdr = Buffer.alloc(8);
      const fd = fs.openSync(this.allocPath, 'r');
      fs.readSync(fd, hdr, 0, 8, 0);
      fs.closeSync(fd);
      const magic = hdr.readUInt32BE(0);
      if (magic !== ALLOC_MAGIC) throw new Error('Invalid alloc file');
    } else {
      const fd = fs.openSync(this.allocPath, 'w');
      const hdr = Buffer.alloc(8);
      hdr.writeUInt32BE(ALLOC_MAGIC, 0);
      hdr.writeUInt32BE(1, 4); // version
      fs.writeSync(fd, hdr);
      fs.fsyncSync(fd);
      fs.closeSync(fd);
    }

    if (fs.existsSync(this.dataPath)) {
      const hdr = Buffer.alloc(12);
      const fd = fs.openSync(this.dataPath, 'r');
      fs.readSync(fd, hdr, 0, 12, 0);
      fs.closeSync(fd);
      this.dataEnd = Number(hdr.readBigUInt64BE(4));
    } else {
      const fd = fs.openSync(this.dataPath, 'w');
      const hdr = Buffer.alloc(12);
      hdr.writeUInt32BE(DATA_MAGIC, 0);
      hdr.writeBigUInt64BE(BigInt(12), 4);
      fs.writeSync(fd, hdr);
      fs.fsyncSync(fd);
      fs.closeSync(fd);
    }
  }

  /** O(1): write tokens at recordId. */
  write(recordId: number, tokens: Token[]): number {
    const [packed, padLen] = packToBytes(tokens);
    const packedBytes = Buffer.from(packed);
    const bitLength = tokens.length * 5;

    // Append to data region
    const dataOffset = this.dataEnd;
    const dataFd = fs.openSync(this.dataPath, 'r+');
    fs.writeSync(dataFd, packedBytes, 0, packedBytes.length, dataOffset);
    fs.fsyncSync(dataFd);
    fs.closeSync(dataFd);

    // Update data end
    this.dataEnd = dataOffset + packedBytes.length;
    const dhFd = fs.openSync(this.dataPath, 'r+');
    const dh = Buffer.alloc(8);
    dh.writeBigUInt64BE(BigInt(this.dataEnd), 0);
    fs.writeSync(dhFd, dh, 0, 8, 4);
    fs.fsyncSync(dhFd);
    fs.closeSync(dhFd);

    // Update alloc entry
    this._writeAllocEntry({ recordId, byteOffset: dataOffset, bitLength, flags: FLAG_ALLOCATED });

    return dataOffset;
  }

  /** Compare-and-swap: atomic check-and-write with cross-process lock. */
  writeIf(recordId: number, tokens: Token[], expectedOffset: number, expectedBitLen: number): boolean {
    // Cross-process lock via exclusive file creation
    const lockPath = this.allocPath + '.cas_lock';
    let fd: number | null = null;
    while (fd === null) {
      try { fd = fs.openSync(lockPath, 'wx'); } catch {}
    }
    try {
      const current = this._readAllocEntry(recordId);
      if (current.byteOffset !== expectedOffset || current.bitLength !== expectedBitLen) return false;
      this.write(recordId, tokens);
      return true;
    } finally {
      try { fs.unlinkSync(lockPath); } catch {}
    }
  }

  /** O(1): read record at recordId. */
  read(recordId: number): AllocRecord | null {
    const entry = this._readAllocEntry(recordId);
    if (entry.flags === FLAG_FREE || entry.flags === FLAG_TOMBSTONE || entry.byteOffset === 0) return null;

    const byteLen = Math.ceil(entry.bitLength / 8);
    const buf = Buffer.alloc(byteLen);
    const fd = fs.openSync(this.dataPath, 'r');
    fs.readSync(fd, buf, 0, byteLen, entry.byteOffset);
    fs.closeSync(fd);

    // Try pad lengths 0-7
    const expectedTokens = Math.floor(entry.bitLength / 5);
    let tokens: Token[] | null = null;
    for (let pad = 0; pad < 8; pad++) {
      try {
        const t = unpackFromBytes(new Uint8Array(buf), pad, expectedTokens);
        if (t.length === expectedTokens) { tokens = t; break; }
      } catch {}
    }
    if (!tokens) return null;

    const parser = new Parser();
    parser.feedTokens(tokens);
    parser.finalize();
    parser.reassemble();

    return {
      recordId,
      tokens,
      parsed: parser.output,
      byteOffset: entry.byteOffset,
      bitLength: entry.bitLength,
      isTombstone: entry.flags === FLAG_TOMBSTONE,
    };
  }

  /** O(1): mark as tombstone. */
  delete(recordId: number): boolean {
    const entry = this._readAllocEntry(recordId);
    if (entry.flags === FLAG_FREE) return false;
    this._writeAllocEntry({ ...entry, flags: FLAG_TOMBSTONE });
    return true;
  }

  /** Compact: remove tombstones, rewrite grid. Returns freed bytes. */
  compact(): number {
    const oldSize = fs.statSync(this.dataPath).size + fs.statSync(this.allocPath).size;
    const tmpDir = fs.mkdtempSync(path.join(this.dataPath, '..', 'compact-'));
    const tmp = new AllocGrid(tmpDir);
    let freed = 0;
    for (let rid = 0; rid < this.totalEntries; rid++) {
      const rec = this.read(rid);
      if (rec && !rec.isTombstone) tmp.write(rid, rec.tokens);
      else if (rec?.isTombstone) freed += Math.ceil(rec.bitLength / 8);
    }
    this.close();
    // Atomic: rename temp files, then commit marker
    for (const fn of ['alloc.grid', 'data.grid']) {
      const src = path.join(tmpDir, fn);
      const dst = path.join(path.dirname(this.allocPath), fn);
      if (fs.existsSync(src)) fs.renameSync(src, dst);
    }
    // Commit marker: if this exists, compaction completed
    const marker = path.join(path.dirname(this.allocPath), 'compact.done');
    fs.writeFileSync(marker, '1'); fs.fsyncSync(fs.openSync(marker, 'r+'));
    fs.unlinkSync(marker);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    this._bootstrap();
    const newSize = fs.statSync(this.dataPath).size + fs.statSync(this.allocPath).size;
    return oldSize - newSize;
  }

  /** Number of allocated entries. */
  get totalEntries(): number {
    const stat = fs.statSync(this.allocPath);
    if (stat.size <= 8) return 0;
    return Math.floor((stat.size - 8) / ALLOC_ENTRY_SIZE);
  }

  get dataFileSize(): number {
    return fs.statSync(this.dataPath).size;
  }

  get allocFileSize(): number {
    return fs.statSync(this.allocPath).size;
  }

  // ── Internals ─────────────────────────────────────────────────────────

  private _readAllocEntry(recordId: number): AllocEntry {
    const off = 8 + recordId * ALLOC_ENTRY_SIZE;
    const buf = Buffer.alloc(ALLOC_ENTRY_SIZE);
    const fd = fs.openSync(this.allocPath, 'r');
    try {
      const bytes = fs.readSync(fd, buf, 0, ALLOC_ENTRY_SIZE, off);
      if (bytes < ALLOC_ENTRY_SIZE) return { recordId, byteOffset: 0, bitLength: 0, flags: FLAG_FREE };
    } catch {
      return { recordId, byteOffset: 0, bitLength: 0, flags: FLAG_FREE };
    } finally {
      fs.closeSync(fd);
    }
    const byteOffset = Number(buf.readBigUInt64BE(0));
    const bitLength = buf.readUInt32BE(8);
    const flags = buf.readUInt32BE(12);
    return { recordId, byteOffset, bitLength, flags };
  }

  private _writeAllocEntry(entry: AllocEntry): void {
    const off = 8 + entry.recordId * ALLOC_ENTRY_SIZE;
    const buf = Buffer.alloc(ALLOC_ENTRY_SIZE);
    buf.writeBigUInt64BE(BigInt(entry.byteOffset), 0);
    buf.writeUInt32BE(entry.bitLength, 8);
    buf.writeUInt32BE(entry.flags, 12);

    // Ensure file is large enough
    const fd = fs.openSync(this.allocPath, 'r+');
    const needed = off + ALLOC_ENTRY_SIZE;
    const stat = fs.statSync(this.allocPath);
    if (stat.size < needed) {
      fs.writeSync(fd, Buffer.alloc(1), 0, 1, needed - 1);
    }
    fs.writeSync(fd, buf, 0, ALLOC_ENTRY_SIZE, off);
    fs.fsyncSync(fd);
    fs.closeSync(fd);
  }

  /** Label-driven reconstruction: join tokens at labeled positions into named fields. */
  static reconstructByLabels(parsed: ParsedToken[]): Record<string, string> {
    const labels: Record<number, string> = {};  // position → label name
    const values: Record<number, string[]> = {}; // position → token texts
    let inLabel = false; let labelPos = 0;
    let dataPos = 0;  // sequential position counter for data tokens

    for (const p of parsed) {
      // Label header detection
      if ((p as any).type === 'command' && (p as any).cmd === 'LABEL') {
        inLabel = true; continue;
      }
      if (inLabel && p.type === 'word') {
        const text = (p as any).text;
        if (text) { labels['_pending'] = text; }  // name before position
        continue;
      }
      if (inLabel && p.type === 'number') {
        const pending = labels['_pending'];
        if (pending) { labels[(p as any).value] = pending; delete labels['_pending']; }
        inLabel = false; continue;
      }
      if (p.type === 'control') continue;

      if (!values[dataPos]) values[dataPos] = [];
      if (p.type === 'number') {
        values[dataPos].push(String((p as any).value));
        dataPos++;  // advance on NUM — each value starts with a number
      } else if (p.type === 'word') {
        values[dataPos].push((p as any).text);
      }
    }

    const result: Record<string, string> = {};
    for (const [pos, name] of Object.entries(labels)) {
      result[name] = (values[Number(pos)] || []).join('');
    }
    return result;
  }

  /** Reconstruct a string from ALL parsed tokens — words + numbers + specials. */
  static reconstructAll(parsed: ParsedToken[]): string {
    let result = '';
    for (const p of parsed) {
      if (p.type === 'word') result += (p as any).text;
      else if (p.type === 'number') result += String((p as any).value);
    }
    return result;
  }

  close(): void {}
  get dataEndVal(): number { return this.dataEnd; }
  set dataEndVal(v: number) { this.dataEnd = v; }
  get allocFPath(): string { return this.allocPath; }
  get dataFPath(): string { return this.dataPath; }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WAL-backed AllocGrid — crash-safe writes via append-only log
// ═══════════════════════════════════════════════════════════════════════════════

const WAL_MAGIC = 0x57414C47; // "WALG"
const WAL_ENTRY_HDR = 16;      // magic(4) + recordId(4) + tokenCount(4) + padLen(4)
const SHA256_LEN = 32;

export class WALedAllocGrid {
  grid: AllocGrid;
  private walPath: string;
  private walSeq = 0;

  constructor(dataDir: string) {
    this.grid = new AllocGrid(dataDir);
    this.walPath = path.join(dataDir, 'wal.grid');
    if (!fs.existsSync(this.walPath)) {
      fs.writeFileSync(this.walPath, Buffer.alloc(0));
    }
    this._replay();
  }

  /** Crash-safe write: WAL with SHA-256 first, then alloc+data. */
  write(recordId: number, tokens: Token[]): number {
    const [packed, padLen] = packToBytes(tokens);
    const packedBytes = Buffer.from(packed);
    const hdr = Buffer.alloc(WAL_ENTRY_HDR);
    hdr.writeUInt32BE(WAL_MAGIC, 0);
    hdr.writeInt32BE(recordId, 4);
    hdr.writeUInt32BE(tokens.length, 8);
    hdr.writeUInt32BE(padLen, 12);
    const body = Buffer.concat([hdr, packedBytes]);
    const hash = crypto.createHash('sha256').update(body).digest();
    const fd = fs.openSync(this.walPath, 'a');
    fs.writeSync(fd, Buffer.concat([body, hash]));
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    this.walSeq++;
    return this.grid.write(recordId, tokens);
  }

  private _replay(): void {
    const data = fs.readFileSync(this.walPath);
    let off = 0;
    while (off + WAL_ENTRY_HDR <= data.length) {
      const magic = data.readUInt32BE(off);
      if (magic !== WAL_MAGIC) break;
      const recordId = data.readInt32BE(off + 4);
      const tokenCount = data.readUInt32BE(off + 8);
      const padLen = data.readUInt32BE(off + 12);
      const tokenBytes = Math.ceil((tokenCount * 5) / 8);
      if (off + WAL_ENTRY_HDR + tokenBytes + SHA256_LEN > data.length) break;
      // Verify SHA-256
      const body = data.subarray(off, off + WAL_ENTRY_HDR + tokenBytes);
      const expected = data.subarray(off + WAL_ENTRY_HDR + tokenBytes, off + WAL_ENTRY_HDR + tokenBytes + SHA256_LEN);
      const computed = crypto.createHash('sha256').update(body).digest();
      if (!computed.equals(expected)) {
        console.error(`WAL SHA-256 mismatch at entry seq ${this.walSeq} — possible corruption`);
        break;
      }
      const tokens = unpackFromBytes(new Uint8Array(data.subarray(off + WAL_ENTRY_HDR, off + WAL_ENTRY_HDR + tokenBytes)), padLen);
      off += WAL_ENTRY_HDR + tokenBytes + SHA256_LEN;
      try { this.grid.write(recordId, tokens); } catch {}
      this.walSeq++;
    }
  }

  read(recordId: number) { return this.grid.read(recordId); }
  delete(recordId: number) { return this.grid.delete(recordId); }
  get totalEntries() { return this.grid.totalEntries; }
  get fileSize() { return this.grid.allocFileSize + this.grid.dataFileSize; }

  /** Checkpoint: flush grid, truncate WAL. */
  checkpoint(): void {
    const bak = this.walPath + '.bak';
    fs.renameSync(this.walPath, bak);
    fs.writeFileSync(this.walPath, '');
    fs.unlinkSync(bak);
    this.walSeq = 0;
  }

  private _replay(): void {
    const data = fs.readFileSync(this.walPath);
    let off = 0;
    while (off + WAL_ENTRY_HDR <= data.length) {
      const magic = data.readUInt32BE(off);
      if (magic !== WAL_MAGIC) break;
      const recordId = data.readInt32BE(off + 4);
      const tokenCount = data.readUInt32BE(off + 8);
      const padLen = data.readUInt32BE(off + 12);
      off += WAL_ENTRY_HDR;
      const tokenBytes = Math.ceil((tokenCount * 5) / 8);
      if (off + tokenBytes > data.length) break;
      const tokens = unpackFromBytes(new Uint8Array(data.subarray(off, off + tokenBytes)), padLen);
      off += tokenBytes;
      // Re-apply
      try { this.grid.write(recordId, tokens); } catch {}
      this.walSeq++;
    }
  }

  close(): void { this.grid.close(); }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group Commit — batched fsync for throughput
// ═══════════════════════════════════════════════════════════════════════════════

interface BufferedWrite {
  recordId: number;
  tokens: Token[];
}

export class GroupCommitAllocGrid {
  private grid: AllocGrid;
  private buffer: BufferedWrite[] = [];
  private batchSize: number;
  private flushCount = 0;

  constructor(dataDir: string, batchSize: number = 50) {
    this.grid = new AllocGrid(dataDir);
    this.batchSize = batchSize;
  }

  /** Buffer a write. Flushes when batch is full. */
  write(recordId: number, tokens: Token[]): void {
    this.buffer.push({ recordId, tokens });
    if (this.buffer.length >= this.batchSize) {
      this.flush();
    }
  }

  /** Force flush all buffered writes — one fsync for entire batch. */
  flush(): void {
    if (this.buffer.length === 0) return;
    // Open FDs once for the entire batch
    const dfd = fs.openSync(this.grid['dataPath'], 'r+');
    const afd = fs.openSync(this.grid['allocFPath'], 'r+');
    let dataEnd = this.grid['dataEnd'];

    for (const w of this.buffer) {
      const [packed, _] = packToBytes(w.tokens);
      const packedBytes = Buffer.from(packed);
      const bitLen = w.tokens.length * 5;
      const off = dataEnd; dataEnd += packedBytes.length;
      // Data
      fs.writeSync(dfd, packedBytes, 0, packedBytes.length, off);
      // Alloc
      const ao = 8 + w.recordId * 16;
      const ab = Buffer.alloc(16);
      ab.writeBigUInt64BE(BigInt(off), 0); ab.writeUInt32BE(bitLen, 8); ab.writeUInt32BE(1, 12);
      if (fs.fstatSync(afd).size < ao + 16) fs.writeSync(afd, Buffer.alloc(1), 0, 1, ao + 15);
      fs.writeSync(afd, ab, 0, 16, ao);
    }
    // Update data header once
    const dh = Buffer.alloc(8); dh.writeBigUInt64BE(BigInt(dataEnd), 0);
    fs.writeSync(dfd, dh, 0, 8, 4);
    // One fsync per file
    fs.fsyncSync(dfd); fs.closeSync(dfd);
    fs.fsyncSync(afd); fs.closeSync(afd);
    this.grid['dataEnd'] = dataEnd;
    this.flushCount++;
    this.buffer = [];
  }

  read(recordId: number) { return this.grid.read(recordId); }
  delete(recordId: number) { return this.grid.delete(recordId); }
  get totalEntries() { return this.grid.totalEntries; }
  get flushes() { return this.flushCount; }
  get pending() { return this.buffer.length; }
  close() { this.flush(); }
}
