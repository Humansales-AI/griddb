/** 5bit RLSEngine — extends AllocGrid, owner-at-position-0. TypeScript. */
import { AllocGrid, Encoder, Token, AllocRecord } from '../index';

export class PermissionDenied extends Error { constructor(m: string) { super(m); this.name = 'PermissionDenied'; } }

export class RLSEngine extends AllocGrid {
  private ownerPos: number; private bypass: number | null = null;
  constructor(dataDir: string, ownerPos = 0) { super(dataDir); this.ownerPos = ownerPos; }

  private getOwner(rid: number): number | null {
    const rec = super.read(rid); if (!rec || rec.isTombstone) return null;
    const vals = rec.parsed.filter(p => p.type === 'number').map((p: any) => p.value);
    return vals.length > this.ownerPos ? vals[this.ownerPos] : (vals[0] || null);
  }

  read(rid: number, uid: number): AllocRecord | null {
    const o = this.getOwner(rid);
    if (o !== null && o !== uid && this.bypass === null) throw new PermissionDenied(`RLS: user ${uid} cannot read record ${rid}`);
    return super.read(rid);
  }

  write(rid: number, uid: number, tokens: Token[]): number {
    const o = this.getOwner(rid);
    if (o !== null && o !== uid && this.bypass === null) throw new PermissionDenied(`RLS: user ${uid} cannot write record ${rid}`);
    return super.write(rid, [...Encoder.encodeInteger(uid), ...tokens]);
  }

  delete(rid: number, uid: number): boolean {
    const o = this.getOwner(rid);
    if (o !== null && o !== uid && this.bypass === null) throw new PermissionDenied(`RLS: user ${uid} cannot delete record ${rid}`);
    return super.delete(rid);
  }

  asAdmin(): RLSEngine { this.bypass = -1; return this; }
  clearAdmin(): void { this.bypass = null; }
}
