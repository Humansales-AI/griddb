/**
 * PositionedGrid — O(1) Bit-Addressed Storage
 * ============================================
 * Record N lives at bit offset N × stride_bits.
 * write(42, tokens) → seek(42 × 1024) → O(1)
 * read(42) → seek(42 × 1024) → read until RECORD → O(1)
 */
import fs from 'fs';
import path from 'path';
import { Token, ParsedToken } from './types';
import { Parser } from './parser';
import { packToBytes, unpackFromBytes } from './serialization';

const DEFAULT_STRIDE_BITS = 1024;
const GRID_MAGIC = 0x47524450; // "GRDP"
const HEADER_SIZE = 12;

export interface PositionedRecord {
  recordId: number; bitOffset: number; tokens: Token[];
  parsed: ParsedToken[]; isTombstone: boolean;
}

export class PositionedGrid {
  strideBits: number;
  strideBytes: number;
  private gridPath: string;
  private totalRows = 0;

  constructor(dataDir: string, strideBits: number = DEFAULT_STRIDE_BITS) {
    this.strideBits = strideBits;
    this.strideBytes = Math.ceil(strideBits / 8);
    fs.mkdirSync(dataDir, { recursive: true });
    this.gridPath = path.join(dataDir, 'main.grid');
    this._bootstrap();
  }

  private _bootstrap(): void {
    if (fs.existsSync(this.gridPath)) {
      const hdr = Buffer.alloc(HEADER_SIZE);
      const fd = fs.openSync(this.gridPath, 'r');
      fs.readSync(fd, hdr, 0, HEADER_SIZE, 0); fs.closeSync(fd);
      const magic = hdr.readUInt32BE(0);
      if (magic === GRID_MAGIC) {
        this.strideBits = hdr.readUInt32BE(4);
        this.strideBytes = Math.ceil(this.strideBits / 8);
        this.totalRows = hdr.readUInt32BE(8);
      }
    } else {
      const fd = fs.openSync(this.gridPath, 'w');
      const hdr = Buffer.alloc(HEADER_SIZE);
      hdr.writeUInt32BE(GRID_MAGIC, 0);
      hdr.writeUInt32BE(this.strideBits, 4);
      hdr.writeUInt32BE(0, 8);
      fs.writeSync(fd, hdr); fs.fsyncSync(fd); fs.closeSync(fd);
    }
  }

  write(recordId: number, tokens: Token[]): number {
    const bitOffset = recordId * this.strideBits;
    const tokenBits = tokens.length * 5;
    if (tokenBits > this.strideBits) throw new Error(`Record exceeds stride: ${tokenBits} > ${this.strideBits}`);
    const [packed, _] = packToBytes(tokens);
    const packedBytes = Buffer.from(packed);
    if (recordId >= this.totalRows) { this.totalRows = recordId + 1; this._updateHeader(); }
    const byteOff = HEADER_SIZE + recordId * this.strideBytes;
    const fd = fs.openSync(this.gridPath, 'r+');
    fs.writeSync(fd, packedBytes, 0, packedBytes.length, byteOff);
    const rem = this.strideBytes - packedBytes.length;
    if (rem > 0) fs.writeSync(fd, Buffer.alloc(rem), 0, rem, byteOff + packedBytes.length);
    fs.fsyncSync(fd); fs.closeSync(fd);
    return bitOffset;
  }

  read(recordId: number): PositionedRecord | null {
    if (recordId >= this.totalRows) return null;
    const byteOff = HEADER_SIZE + recordId * this.strideBytes;
    const raw = Buffer.alloc(this.strideBytes);
    const fd = fs.openSync(this.gridPath, 'r');
    fs.readSync(fd, raw, 0, this.strideBytes, byteOff); fs.closeSync(fd);
    // Trim trailing nulls
    let end = raw.length;
    while (end > 0 && raw[end - 1] === 0) end--;
    if (end === 0) return null;
    const trimmed = raw.subarray(0, end);
    // Try pad lengths
    for (let pad = 0; pad < 8; pad++) {
      try {
        const t = unpackFromBytes(new Uint8Array(trimmed), pad);
        if (t.length > 0 && t[t.length - 1] === Token.RECORD) {
          const p = new Parser(); p.feedTokens(t); p.finalize();
          return { recordId, bitOffset: recordId * this.strideBits, tokens: t,
                   parsed: p.output, isTombstone: t.length === 3 && t[0] === Token.D0 && t[2] === Token.RECORD };
        }
      } catch {}
    }
    return null;
  }

  delete(recordId: number): boolean {
    this.write(recordId, [Token.D0, Token.END, Token.RECORD]); return true;
  }

  scan(start: number, end?: number): PositionedRecord[] {
    const r: PositionedRecord[] = [];
    for (let i = start; i < (end ?? this.totalRows); i++) {
      const rec = this.read(i);
      if (rec && !rec.isTombstone) r.push(rec);
    }
    return r;
  }

  private _updateHeader(): void {
    const hdr = Buffer.alloc(HEADER_SIZE);
    hdr.writeUInt32BE(GRID_MAGIC, 0); hdr.writeUInt32BE(this.strideBits, 4); hdr.writeUInt32BE(this.totalRows, 8);
    const fd = fs.openSync(this.gridPath, 'r+'); fs.writeSync(fd, hdr, 0, HEADER_SIZE, 0); fs.fsyncSync(fd); fs.closeSync(fd);
  }

  get rows(): number { return this.totalRows; }
  get filePath(): string { return this.gridPath; }
  close(): void {}
}
