/**
 * SingleFileGrid — One grid.db like SQLite
 * ==========================================
 * Everything in one file:
 *   [Header 64B] [Alloc Table] [Data Blobs] [WAL Entries...]
 *
 * Same logic as AllocGrid + WALedAllocGrid, one fopen().
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { Token, ParsedToken, ParsedNumber } from './types';
import { Parser } from './parser';
import { packToBytes, unpackFromBytes } from './serialization';

const MAGIC = 0x47524442; // "GRDB"
const VERSION = 1;
const HEADER_SIZE = 64;
const ALLOC_ENTRY = 16;

interface AllocEntry {
  offset: number; bitLength: number; flags: number; // 0=free, 1=alloc, 2=tombstone
}

export interface SingleFileRecord {
  recordId: number; tokens: Token[]; parsed: ParsedToken[];
  byteOffset: number; bitLength: number; isTombstone: boolean;
}

export class SingleFileGrid {
  private fd!: number;
  private filePath: string;
  private allocStart = HEADER_SIZE;
  private allocCount = 0;
  private dataStart = HEADER_SIZE;
  private dataEnd = HEADER_SIZE;
  private walEnd = HEADER_SIZE;

  constructor(filePath: string) {
    this.filePath = filePath;
    const exists = fs.existsSync(filePath);
    this.fd = fs.openSync(filePath, exists ? 'r+' : 'w+');
    if (!exists) {
      this._writeHeader(0, 0, 0);
    } else {
      this._readHeader();
      this._replayWAL();
    }
  }

  // ── Header ────────────────────────────────────────────────────────────

  private _writeHeader(allocSize: number, dataSize: number, walSize: number): void {
    this.allocCount = Math.floor(allocSize / ALLOC_ENTRY);
    this.dataStart = this.allocStart + allocSize;
    this.dataEnd = this.dataStart + dataSize;
    this.walEnd = this.dataEnd;

    const h = Buffer.alloc(HEADER_SIZE);
    h.writeUInt32BE(MAGIC, 0); h.writeUInt32BE(VERSION, 4);
    h.writeBigUInt64BE(BigInt(allocSize), 8);
    h.writeBigUInt64BE(BigInt(dataSize), 16);
    h.writeBigUInt64BE(BigInt(walSize), 24);
    fs.writeSync(this.fd, h, 0, HEADER_SIZE, 0);
    fs.fsyncSync(this.fd);
  }

  private _readHeader(): void {
    const h = Buffer.alloc(HEADER_SIZE);
    fs.readSync(this.fd, h, 0, HEADER_SIZE, 0);
    const magic = h.readUInt32BE(0);
    if (magic !== MAGIC) throw new Error('Invalid grid.db file');
    const allocSize = Number(h.readBigUInt64BE(8));
    const dataSize = Number(h.readBigUInt64BE(16));
    const walSize = Number(h.readBigUInt64BE(24));
    this.allocCount = Math.floor(allocSize / ALLOC_ENTRY);
    this.dataStart = this.allocStart + allocSize;
    this.dataEnd = this.dataStart + dataSize;
    this.walEnd = this.dataEnd + walSize;
  }

  private _updateSizes(allocSize: number, dataSize: number, walSize: number): void {
    const h = Buffer.alloc(24);
    h.writeBigUInt64BE(BigInt(allocSize), 0);
    h.writeBigUInt64BE(BigInt(dataSize), 8);
    h.writeBigUInt64BE(BigInt(walSize), 16);
    fs.writeSync(this.fd, h, 0, 24, 8);
    fs.fsyncSync(this.fd);
    this.allocCount = Math.floor(allocSize / ALLOC_ENTRY);
    this.dataStart = this.allocStart + allocSize;
    this.dataEnd = this.dataStart + dataSize;
    this.walEnd = this.dataEnd + walSize;
  }

  // ── Alloc Table ──────────────────────────────────────────────────────

  private _readAlloc(recordId: number): AllocEntry {
    const off = this.allocStart + recordId * ALLOC_ENTRY;
    const b = Buffer.alloc(ALLOC_ENTRY);
    try {
      fs.readSync(this.fd, b, 0, ALLOC_ENTRY, off);
      return {
        offset: Number(b.readBigUInt64BE(0)),
        bitLength: b.readUInt32BE(8),
        flags: b.readUInt32BE(12),
      };
    } catch { return { offset: 0, bitLength: 0, flags: 0 }; }
  }

  private _writeAlloc(recordId: number, entry: AllocEntry): void {
    const off = this.allocStart + recordId * ALLOC_ENTRY;
    const needed = off + ALLOC_ENTRY;
    // Grow file if needed
    const stat = fs.fstatSync(this.fd);
    if (stat.size < needed) {
      fs.ftruncateSync(this.fd, needed);
    }
    const b = Buffer.alloc(ALLOC_ENTRY);
    b.writeBigUInt64BE(BigInt(entry.offset), 0);
    b.writeUInt32BE(entry.bitLength, 8);
    b.writeUInt32BE(entry.flags, 12);
    fs.writeSync(this.fd, b, 0, ALLOC_ENTRY, off);
    fs.fsyncSync(this.fd);
    if (recordId >= this.allocCount) {
      this._updateSizes((recordId + 1) * ALLOC_ENTRY, this.dataEnd - this.dataStart, 0);
    }
  }

  // ── Core O(1) Operations ─────────────────────────────────────────────

  write(recordId: number, tokens: Token[]): number {
    // 1. Write to WAL (append at end of data+WAL region)
    const [packed, padLen] = packToBytes(tokens);
    const packedBytes = Buffer.from(packed);
    const bitLen = tokens.length * 5;

    // WAL entry: [magic(4)] [recordId(4)] [tokenCount(4)] [padLen(4)] [data...]
    const walHdr = Buffer.alloc(16);
    walHdr.writeUInt32BE(0x57414C47, 0); // "WALG"
    walHdr.writeInt32BE(recordId, 4);
    walHdr.writeUInt32BE(tokens.length, 8);
    walHdr.writeUInt32BE(padLen, 12);
    const walEntry = Buffer.concat([walHdr, packedBytes]);
    fs.writeSync(this.fd, walEntry, 0, walEntry.length, this.walEnd);
    fs.fsyncSync(this.fd);
    this.walEnd += walEntry.length;

    // 2. Apply to data + alloc
    const dataOff = this.dataEnd;
    fs.writeSync(this.fd, packedBytes, 0, packedBytes.length, this.dataEnd);
    this.dataEnd += packedBytes.length;

    this._writeAlloc(recordId, { offset: dataOff, bitLength: bitLen, flags: 1 });

    const allocSize = this.allocCount * ALLOC_ENTRY;
    const dataSize = this.dataEnd - this.dataStart;
    this._updateSizes(allocSize, dataSize, 0);

    return dataOff;
  }

  read(recordId: number): SingleFileRecord | null {
    const ae = this._readAlloc(recordId);
    if (ae.flags === 0 || ae.offset === 0) return null;
    const byteLen = Math.ceil(ae.bitLength / 8);
    const buf = Buffer.alloc(byteLen);
    fs.readSync(this.fd, buf, 0, byteLen, ae.offset);

    const expected = Math.floor(ae.bitLength / 5);
    for (let pad = 0; pad < 8; pad++) {
      try {
        const t = unpackFromBytes(new Uint8Array(buf), pad, expected);
        if (t.length === expected) {
          const p = new Parser(); p.feedTokens(t); p.finalize();
          return { recordId, tokens: t, parsed: p.output, byteOffset: ae.offset, bitLength: ae.bitLength, isTombstone: ae.flags === 2 };
        }
      } catch {}
    }
    return null;
  }

  delete(recordId: number): boolean {
    const ae = this._readAlloc(recordId);
    if (ae.flags === 0) return false;
    this._writeAlloc(recordId, { ...ae, flags: 2 });
    return true;
  }

  // ── WAL Replay ──────────────────────────────────────────────────────

  private _replayWAL(): void {
    // WAL entries live between dataEnd and walEnd
    let off = this.dataEnd; // WAL starts right after data
    const stat = fs.fstatSync(this.fd);
    const fileEnd = stat.size;

    // Re-read header to get accurate sizes
    this._readHeader();
    off = this.dataEnd;

    while (off + 16 <= fileEnd) {
      const h = Buffer.alloc(16);
      try { fs.readSync(this.fd, h, 0, 16, off); } catch { break; }
      const magic = h.readUInt32BE(0);
      if (magic !== 0x57414C47) break;
      const recordId = h.readInt32BE(4);
      const tokenCount = h.readUInt32BE(8);
      const padLen = h.readUInt32BE(12);
      off += 16;
      const tokenBytes = Math.ceil((tokenCount * 5) / 8);
      if (off + tokenBytes > fileEnd) break;
      const tb = Buffer.alloc(tokenBytes);
      fs.readSync(this.fd, tb, 0, tokenBytes, off);
      off += tokenBytes;
      const tokens = unpackFromBytes(new Uint8Array(tb), padLen);
      // Apply
      const [packed, _] = packToBytes(tokens);
      const packedBytes = Buffer.from(packed);
      const dataOff = this.dataEnd;
      fs.writeSync(this.fd, packedBytes, 0, packedBytes.length, dataOff);
      this.dataEnd += packedBytes.length;
      this._writeAlloc(recordId, { offset: dataOff, bitLength: tokenCount * 5, flags: 1 });
    }

    // Truncate WAL after replay
    if (off < fileEnd) { fs.ftruncateSync(this.fd, off); }
    this.walEnd = this.dataEnd;
  }

  // ── Stats ────────────────────────────────────────────────────────────

  get totalEntries(): number { return this.allocCount; }
  get fileSize(): number { return fs.fstatSync(this.fd).size; }

  scan(max: number = 100): SingleFileRecord[] {
    const r: SingleFileRecord[] = [];
    for (let i = 0; i < this.allocCount && r.length < max; i++) {
      const rec = this.read(i);
      if (rec && !rec.isTombstone) r.push(rec);
    }
    return r;
  }

  close(): void { fs.closeSync(this.fd); }
}
