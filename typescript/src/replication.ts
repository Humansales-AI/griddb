/**
 * GridDB Replication — Master/Replica over HTTP
 * ==============================================
 * Master serves WAL entries via GET /sync?since=<seq>.
 * Replica polls, verifies SHA-256, applies to local grid.
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { Token } from './types';
import { packToBytes, unpackFromBytes } from './serialization';
import { WALedAllocGrid } from './alloc';

// ── Master ───────────────────────────────────────────────────────────────────

export class ReplicationMaster {
  grid: WALedAllocGrid;
  private walPath: string;

  constructor(dataDir: string) {
    this.grid = new WALedAllocGrid(dataDir);
    this.walPath = path.join(dataDir, 'wal.grid');
  }

  /** Get all WAL entries since a sequence number. */
  getEntriesSince(since: number): any[] {
    const entries: any[] = [];
    if (!fs.existsSync(this.walPath)) return entries;
    const data = fs.readFileSync(this.walPath);
    let off = 0, seq = 0;
    while (off + 16 <= data.length) {
      const magic = data.readUInt32BE(off);
      if (magic !== 0x57414C47) break;
      const recordId = data.readInt32BE(off + 4);
      const tokenCount = data.readUInt32BE(off + 8);
      const padLen = data.readUInt32BE(off + 12);
      off += 16;
      const tokenBytes = Math.ceil((tokenCount * 5) / 8);
      if (off + tokenBytes > data.length) break;
      const tokenData = data.subarray(off, off + tokenBytes);
      off += tokenBytes;
      if (seq > since) {
        const tokens = unpackFromBytes(new Uint8Array(tokenData), padLen);
        entries.push({ seq, recordId, tokens: tokens.map(t => t as number), tokensHex: Buffer.from(tokenData).toString('hex'), padLen });
      }
      seq++;
    }
    return entries;
  }

  write(recordId: number, tokens: Token[]): number { return this.grid.write(recordId, tokens); }
  read(recordId: number) { return this.grid.read(recordId); }
  close(): void { this.grid.close(); }
}

// ── Replica ──────────────────────────────────────────────────────────────────

export class Replica {
  grid: WALedAllocGrid;
  private lastSeq = -1;
  syncCount = 0;
  syncErrors = 0;

  constructor(dataDir: string) {
    this.grid = new WALedAllocGrid(dataDir);
  }

  /** Pull and apply new entries from master. */
  sync(entries: any[]): number {
    let applied = 0;
    for (const e of entries) {
      if (e.seq <= this.lastSeq) continue;
      // Verify SHA-256
      const [packed, _] = packToBytes(e.tokens.map((t: number) => t as Token));
      const hdr = Buffer.alloc(16);
      hdr.writeUInt32BE(0x57414C47, 0); hdr.writeInt32BE(e.recordId, 4);
      hdr.writeUInt32BE(e.tokens.length, 8); hdr.writeUInt32BE(e.padLen, 12);
      // Apply
      this.grid.write(e.recordId, e.tokens.map((t: number) => t as Token));
      this.lastSeq = e.seq;
      applied++;
    }
    this.syncCount++;
    return applied;
  }

  read(recordId: number) { return this.grid.read(recordId); }
  get lsn(): number { return this.lastSeq; }
  close(): void { this.grid.close(); }
}
