/**
 * 5bit CommandRLS — SPECIAL3 AUTH/GRANT/REVOKE tokens (TypeScript)
 * =================================================================
 * Permission representation in the token fabric.
 * Extends AllocGrid. Owner-gated grant/revoke.
 */
import { AllocGrid, Encoder, Token, Parser, ParsedNumber, AllocRecord } from '../index';

const CMD_AUTH = Token.D0; const CMD_GRANT_R = Token.D1;
const CMD_REVOKE = Token.D3;
const CMD_TOKENS = new Set([CMD_AUTH, CMD_GRANT_R, CMD_REVOKE]);

export class PermissionDenied extends Error { constructor(m: string) { super(m); this.name='PermissionDenied'; } }

export class CommandRLS extends AllocGrid {
  private bypass: number | null = null;

  constructor(dataDir: string) { super(dataDir); }

  writeOwned(rid: number, uid: number, tokens: Token[]): number {
    const existing = super.read(rid);
    if (existing && !existing.isTombstone) {
      const owner = this.getOwner(existing.tokens);
      if (owner !== null && owner !== uid && this.bypass === null) {
        throw new PermissionDenied(`Cannot overwrite record ${rid}`);
      }
    }
    const cmd = Encoder.encodeInteger(uid); // AUTH token via START×4 + CMD_AUTH
    return super.write(rid, [...cmd, ...tokens]);
  }

  read(rid: number, uid?: number): AllocRecord | null {
    const rec = super.read(rid); if (!rec || rec.isTombstone) return null;
    if (uid !== undefined) {
      const owner = this.getOwner(rec.tokens);
      const grantees = this.getGrantees(rec.tokens);
      if (owner !== null && owner !== uid && !grantees.has(uid) && this.bypass === null) {
        throw new PermissionDenied(`User ${uid} cannot read record ${rid}`);
      }
    }
    return rec;
  }

  grantRead(rid: number, ownerUid: number, grantee: number): void {
    const rec = super.read(rid); if (!rec) return;
    const owner = this.getOwner(rec.tokens);
    if (owner !== ownerUid && this.bypass === null) throw new PermissionDenied('Only owner can grant');
    const tokens = [...rec.tokens, ...Encoder.encodeInteger(grantee), Token.RECORD];
    super.write(rid, tokens);
  }

  revokeRead(rid: number, ownerUid: number, target: number): void {
    const rec = super.read(rid); if (!rec) return;
    const owner = this.getOwner(rec.tokens);
    if (owner !== ownerUid && this.bypass === null) throw new PermissionDenied('Only owner can revoke');
    // Rebuild without grantee token
    const nums = rec.tokens.filter(t => typeof t === 'number');
    super.write(rid, nums as Token[]);
  }

  private getOwner(tokens: Token[]): number | null {
    const p = new Parser(); for (const t of tokens) p.feed(t); p.finalize();
    const nums = p.output.filter(x => x.type === 'number').map((x: any) => x.value);
    return nums.length > 0 ? nums[0] : null;
  }

  private getGrantees(tokens: Token[]): Set<number> {
    const p = new Parser(); for (const t of tokens) p.feed(t); p.finalize();
    const nums = p.output.filter(x => x.type === 'number').map((x: any) => x.value);
    return new Set(nums.slice(1));
  }

  asAdmin(): CommandRLS { this.bypass = -1; return this; }
}
