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
import { packToBytes } from './serialization';
import { WALedAllocGrid } from './alloc';
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
  private dataDir: string;
  syncCount = 0;
  syncErrors = 0;

  constructor(dataDir: string) {
    this.dataDir = dataDir;
    this.grid = new WALedAllocGrid(dataDir);
    // Load persisted master cursor (Bug B fix)
    if (!this._loadCursor()) {
      this._lastMasterSeq = -1;
    }
  }

  private _lastMasterSeq = -1;
  private _cursorPath() { return path.join(this.dataDir, 'replica.cursor'); }

  private _loadCursor(): boolean {
    try {
      const data = fs.readFileSync(this._cursorPath(), 'utf-8');
      this._lastMasterSeq = parseInt(data.trim());
      return true;
    } catch { return false; }
  }

  private _saveCursor(): void {
    fs.writeFileSync(this._cursorPath(), String(this._lastMasterSeq));
    fs.fsyncSync(fs.openSync(this._cursorPath(), 'r+'));
  }

  /** Pull and apply new entries from master. Halts on SHA failure — no holes. */
  sync(entries: any[]): number {
    let applied = 0;
    for (const e of entries) {
      if (e.seq <= this._lastMasterSeq) continue;

      // Verify SHA-256
      if (e.tokensHex) {
        const [packed, _] = packToBytes(e.tokens.map((t: number) => t as Token));
        const computed = crypto.createHash('sha256').update(Buffer.from(packed)).digest('hex');
        if (computed !== e.tokensHex) {
          this.syncErrors++;
          console.error(`Replica: SHA mismatch at seq ${e.seq} — halting sync, cursor stays at ${this._lastMasterSeq}`);
          return applied;  // Bug A fix: halt, don't skip
        }
      }

      this.grid.write(e.recordId, e.tokens.map((t: number) => t as Token));
      this._lastMasterSeq = e.seq;
      applied++;
    }
    this._saveCursor();  // Bug B fix: persist master cursor separately
    this.syncCount++;
    return applied;
  }

  read(recordId: number) { return this.grid.read(recordId); }
  get lsn(): number { return this._lastMasterSeq; }
  close(): void { this.grid.close(); }
}
