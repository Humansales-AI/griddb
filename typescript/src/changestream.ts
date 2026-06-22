/**
 * GridDB Change Streams — Live WAL Events
 * =======================================
 * Tail the WAL, emit structured events to subscribers.
 * Supports SSE, long-poll, filtering by recordId.
 */
import fs from 'fs';
import path from 'path';
import { Token } from './types';
import { unpackFromBytes } from './serialization';
import { WALedAllocGrid } from './alloc';

export interface ChangeEvent {
  seq: number; type: string; recordId: number; txnId: number;
  tokens: number[]; timestamp: number;
}

export class ChangeStream {
  private walPath: string;
  private lastSeq = -1;
  private subscribers: Array<(event: ChangeEvent) => void> = [];
  private _running = false;
  private _interval: any = null;

  constructor(dataDir: string) {
    this.walPath = path.join(dataDir, 'wal.grid');
  }

  /** Start tailing WAL in background. */
  start(pollMs: number = 500): void {
    this._running = true;
    this._interval = setInterval(() => this._poll(), pollMs);
  }

  stop(): void {
    this._running = false;
    if (this._interval) clearInterval(this._interval);
  }

  subscribe(fn: (event: ChangeEvent) => void): void { this.subscribers.push(fn); }

  /** Get all events since a sequence number. */
  getEventsSince(since: number, filterRid?: number): ChangeEvent[] {
    const events: ChangeEvent[] = [];
    if (!fs.existsSync(this.walPath)) return events;
    const data = fs.readFileSync(this.walPath);
    let off = 0, seq = 0;
    while (off + 16 <= data.length) {
      const magic = data.readUInt32BE(off);
      if (magic !== 0x57414C47) break;
      const rid = data.readInt32BE(off + 4);
      const tc = data.readUInt32BE(off + 8);
      const pad = data.readUInt32BE(off + 12);
      off += 16;
      const tb = Math.ceil((tc * 5) / 8);
      if (off + tb > data.length) break;
      const t = unpackFromBytes(new Uint8Array(data.subarray(off, off + tb)), pad);
      off += tb;
      if (seq > since && (!filterRid || rid === filterRid)) {
        events.push({ seq, type: 'PUT', recordId: rid, txnId: 0, tokens: t.map(x => x as number), timestamp: Date.now() });
      }
      seq++;
    }
    return events;
  }

  private _poll(): void {
    const newEvents = this.getEventsSince(this.lastSeq);
    for (const e of newEvents) {
      this.lastSeq = e.seq;
      for (const sub of this.subscribers) { try { sub(e); } catch {} }
    }
  }
}
