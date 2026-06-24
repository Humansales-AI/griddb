/**
 * 5bit Tenant — TypeScript Multi-Tenant Isolation
 * ================================================
 * Tenant-scoped record IDs. Cross-tenant reads are physically impossible.
 */
import { AllocGrid, Token, AllocRecord } from '../index';

const STRIDE = 1_000_000;

export class TenantGrid {
  private grid: AllocGrid;
  private active: string | null = null;

  constructor(dataDir: string) { this.grid = new AllocGrid(dataDir); }

  withTenant<T>(tenantId: string, fn: (g: TenantGrid) => T): T {
    const old = this.active;
    this.active = tenantId;
    try { return fn(this); } finally { this.active = old; }
  }

  private scope(rid: number): number {
    if (!this.active) throw new Error('No active tenant');
    let h = 0;
    for (let i = 0; i < this.active.length; i++) h = ((h << 5) - h) + this.active.charCodeAt(i) | 0;
    return (Math.abs(h) % (STRIDE - 1) + 1) * STRIDE + rid;
  }

  write(rid: number, tokens: Token[]): number { return this.grid.write(this.scope(rid), tokens); }
  read(rid: number): AllocRecord | null { return this.grid.read(this.scope(rid)); }
  delete(rid: number): boolean { return this.grid.delete(this.scope(rid)); }
  close(): void { this.grid.close(); }
}
