/**
 * 5bit PerUserGrid — True Multi-Tenant Isolation (TypeScript)
 * =============================================================
 * Per-user password keys. No shared secret.
 * Server cannot read user data without their password.
 */
import crypto from 'crypto';
import { AllocGrid, Encoder, Token, Parser, ParsedNumber, packToBytes, unpackFromBytes, AllocRecord } from '../index';

export class PerUserGrid extends AllocGrid {
  private currentKey: Buffer | null = null;
  private keyCache: Map<number, Buffer> = new Map();

  unlock(userId: number, password: string): void {
    const salt = `5bit-user-${userId}`;
    this.currentKey = crypto.pbkdf2Sync(password, salt, 200_000, 32, 'sha256');
    this.keyCache.set(userId, this.currentKey);
  }

  lock(): void { this.currentKey = null; this.keyCache.clear(); }

  private requireKey(): void {
    if (!this.currentKey) throw new Error('Call unlock(user_id, password) first');
  }

  write(recordId: number, tokens: Token[]): number {
    this.requireKey();
    const info = Buffer.alloc(8); info.writeBigUInt64BE(BigInt(recordId));
    const recordKey = crypto.createHmac('sha256', this.currentKey!).update(Buffer.concat([info, Buffer.from([1])])).digest();
    const encKey = crypto.createHmac('sha256', recordKey).update('enc').digest();
    const macKey = crypto.createHmac('sha256', recordKey).update('mac').digest();

    const [packed, pad] = packToBytes(tokens);
    const nonce = crypto.randomBytes(16);
    const ct = this.ctr(encKey, nonce, Buffer.from(packed));
    const mac = crypto.createHmac('sha256', macKey).update(ct).digest().subarray(0, 16);

    const blob = Buffer.concat([Buffer.from([pad]), nonce, ct, mac]);
    const blobTokens: Token[] = [];
    for (const b of blob) blobTokens.push(...Encoder.encodeInteger(b));
    blobTokens.push(Token.RECORD);
    return super.write(recordId, blobTokens);
  }

  read(recordId: number): AllocRecord | null {
    this.requireKey();
    const rec = super.read(recordId);
    if (!rec || rec.isTombstone) return null;

    const nums = rec.parsed.filter(p => p.type === 'number').map((p: any) => p.value);
    if (nums.length < 18) throw new Error('Corrupted');
    const blob = Buffer.from(nums);
    const pad = blob[0], nonce = blob.subarray(1, 17), mac = blob.subarray(blob.length - 16), ct = blob.subarray(17, blob.length - 16);

    const info = Buffer.alloc(8); info.writeBigUInt64BE(BigInt(recordId));
    const recordKey = crypto.createHmac('sha256', this.currentKey!).update(Buffer.concat([info, Buffer.from([1])])).digest();
    const macKey = crypto.createHmac('sha256', recordKey).update('mac').digest();
    const expected = crypto.createHmac('sha256', macKey).update(ct).digest().subarray(0, 16);
    if (!crypto.timingSafeEqual(mac, expected)) throw new Error('Wrong key');

    const encKey = crypto.createHmac('sha256', recordKey).update('enc').digest();
    const pt = this.ctr(encKey, nonce, ct);
    const tokens = unpackFromBytes(new Uint8Array(pt), pad);
    const p = new Parser(); p.feedTokens(tokens); p.finalize();
    if (typeof (p as any).reassemble === 'function') (p as any).reassemble();
    return { recordId, tokens, parsed: p.output, byteOffset: rec.byteOffset,
             bitLength: tokens.length * 5, isTombstone: false, valueVector: [], digitVector: [], parsedValues: [] };
  }

  delete(recordId: number): boolean { this.read(recordId); return super.delete(recordId); }

  private ctr(key: Buffer, nonce: Buffer, data: Buffer): Buffer {
    const r = Buffer.alloc(data.length);
    for (let i = 0; i < data.length; i += 32) {
      const ctr = Buffer.concat([nonce, Buffer.from([0,0,0,0,0,0,0,0])]);
      ctr.writeBigUInt64BE(BigInt(Math.floor(i/32)), 16);
      const ks = crypto.createHmac('sha256', key).update(ctr).digest();
      for (let j = i; j < Math.min(i+32, data.length); j++) r[j] = data[j] ^ ks[j-i];
    }
    return r;
  }
}
