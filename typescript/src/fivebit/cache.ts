/** 5bit Page Cache — LRU read cache. TypeScript. */
import { AllocGrid, Token, AllocRecord } from '../index';

export class CachedGrid {
  private grid: AllocGrid;
  private cache: Map<number, AllocRecord> = new Map();
  private order: number[] = [];
  private maxSize: number;
  hits = 0; misses = 0;

  constructor(dataDir: string, cacheSize = 1000) {
    this.grid = new AllocGrid(dataDir);
    this.maxSize = cacheSize;
  }

  read(recordId: number): AllocRecord | null {
    if (this.cache.has(recordId)) {
      this.order = this.order.filter(id => id !== recordId);
      this.order.push(recordId);
      this.hits++;
      return this.cache.get(recordId)!;
    }
    const rec = this.grid.read(recordId);
    if (rec) {
      if (this.cache.size >= this.maxSize) {
        const evict = this.order.shift()!;
        this.cache.delete(evict);
      }
      this.cache.set(recordId, rec);
      this.order.push(recordId);
      this.misses++;
    }
    return rec;
  }

  write(rid: number, tokens: Token[]): number {
    this.cache.delete(rid);
    this.order = this.order.filter(id => id !== rid);
    return this.grid.write(rid, tokens);
  }

  delete(rid: number): boolean {
    this.cache.delete(rid);
    this.order = this.order.filter(id => id !== rid);
    return this.grid.delete(rid);
  }

  get hitRate(): number { const t = this.hits + this.misses; return t ? this.hits / t : 0; }
  get totalEntries(): number { return this.grid.totalEntries; }
  close(): void { this.grid.close(); }
}
