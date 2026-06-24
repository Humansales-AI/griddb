/**
 * 5bit MultiMode Auth — TypeScript (mode='zero' | 'managed')
 * ============================================================
 * mode='zero'    → PerUserGrid, server blind, recovery codes
 * mode='managed' → shared-secret, server can read, password reset
 */
import crypto from 'crypto';
import { AllocGrid, Encoder, Token, Parser, ParsedNumber, ParsedWord, packToBytes, unpackFromBytes, AllocRecord } from '../index';
import type { PerUserGrid } from './per_user';

const MODE_BASE = 80_000_000;
const WORDS = ['abandon','ability','able','about','above','absent','absorb','abstract','acoustic','acquire','across','act','action','actor','actress'];

export class MultiModeGrid {
  private base: AllocGrid;
  private appSecret: Buffer;
  private currentKey: Buffer | null = null;
  private currentUser: number | null = null;
  private currentMode: string | null = null;
  private modeCache: Map<number, string> = new Map();

  constructor(dataDir: string, appSecret?: Buffer) {
    this.base = new AllocGrid(dataDir);
    this.appSecret = appSecret || crypto.randomBytes(32);
  }

  signup(userId: number, password: string, mode: string = 'zero'): void {
    if (mode !== 'zero' && mode !== 'managed') throw new Error("mode must be 'zero' or 'managed'");
    this.base.write(MODE_BASE + userId, [
      ...Encoder.encodeInteger(userId), ...Encoder.encodeWord(mode.toUpperCase()), Token.RECORD,
    ]);
    this.modeCache.set(userId, mode);
    const salt = `5bit-user-${userId}`;
    const userKey = crypto.pbkdf2Sync(password, salt, 200_000, 32, 'sha256');
    if (mode === 'zero') {
      const recovery = this.genRecovery(userKey);
      this.base.write(MODE_BASE + userId + 1, [...Encoder.encodeWord(recovery), Token.RECORD]);
    } else {
      const nonce = crypto.randomBytes(16);
      const ct = this.ctr(this.appSecret, nonce, userKey);
      const mac = crypto.createHmac('sha256', this.appSecret).update(ct).digest().subarray(0, 16);
      const blob = Buffer.concat([nonce, ct, mac]);
      const bt: Token[] = []; for (const b of blob) bt.push(...Encoder.encodeInteger(b));
      bt.push(Token.RECORD); this.base.write(MODE_BASE + userId + 2, bt);
    }
  }

  login(userId: number, password: string): void {
    const salt = `5bit-user-${userId}`;
    this.currentKey = crypto.pbkdf2Sync(password, salt, 200_000, 32, 'sha256');
    this.currentUser = userId;
    this.currentMode = this.getMode(userId);
  }

  lock(): void { this.currentKey = null; this.currentUser = null; this.currentMode = null; }

  getMode(userId: number): string {
    if (this.modeCache.has(userId)) return this.modeCache.get(userId)!;
    const rec = this.base.read(MODE_BASE + userId);
    if (rec) {
      const words = rec.parsed.filter(p => p.type === 'word').map((p: any) => p.text);
      const m = words.join('').toLowerCase();
      this.modeCache.set(userId, m);
      return m;
    }
    return 'zero';
  }

  getRecoveryCode(userId: number): string | null {
    const rec = this.base.read(MODE_BASE + userId + 1);
    if (rec) return rec.parsed.filter(p => p.type === 'word').map((p: any) => p.text).join('');
    return null;
  }

  write(recordId: number, tokens: Token[]): number {
    if (!this.currentKey) throw new Error('Call login() first');
    const info = Buffer.alloc(8); info.writeBigUInt64BE(BigInt(recordId));
    const key = this.currentMode === 'zero' ? this.currentKey : this.appSecret;
    const recordKey = crypto.createHmac('sha256', key).update(Buffer.concat([info, Buffer.from([1])])).digest();
    const encKey = crypto.createHmac('sha256', recordKey).update('enc').digest();
    const macKey = crypto.createHmac('sha256', recordKey).update('mac').digest();
    const [packed, pad] = packToBytes(tokens);
    const nonce = crypto.randomBytes(16);
    const ct = this.ctr(encKey, nonce, Buffer.from(packed));
    const mac = crypto.createHmac('sha256', macKey).update(ct).digest().subarray(0, 16);
    const blob = Buffer.concat([Buffer.from([pad]), nonce, ct, mac]);
    const bt: Token[] = []; for (const b of blob) bt.push(...Encoder.encodeInteger(b));
    bt.push(Token.RECORD); return this.base.write(recordId, bt);
  }

  read(recordId: number): AllocRecord | null {
    if (!this.currentKey) throw new Error('Call login() first');
    const rec = this.base.read(recordId);
    if (!rec || rec.flags === 2) return null;
    const nums = rec.parsed.filter(p => p.type === 'number').map((p: any) => p.value);
    if (nums.length < 18) throw new Error('Corrupted');
    const blob = Buffer.from(nums), pad = blob[0], nonce = blob.subarray(1,17), mac = blob.subarray(blob.length-16), ct = blob.subarray(17,blob.length-16);
    const info = Buffer.alloc(8); info.writeBigUInt64BE(BigInt(recordId));
    const key = this.currentMode === 'zero' ? this.currentKey : this.appSecret;
    const recordKey = crypto.createHmac('sha256', key).update(Buffer.concat([info, Buffer.from([1])])).digest();
    const macKey = crypto.createHmac('sha256', recordKey).update('mac').digest();
    if (!crypto.timingSafeEqual(mac, crypto.createHmac('sha256', macKey).update(ct).digest().subarray(0,16))) throw new Error('Wrong key');
    const encKey = crypto.createHmac('sha256', recordKey).update('enc').digest();
    const pt = this.ctr(encKey, nonce, ct);
    const tokens = unpackFromBytes(new Uint8Array(pt), pad);
    const p = new Parser(); p.feedTokens(tokens); p.finalize();
    if (typeof (p as any).reassemble === 'function') (p as any).reassemble();
    return { recordId, tokens, parsed: p.output, byteOffset: rec.byteOffset,
             bitLength: tokens.length*5, isTombstone: false, valueVector: [], digitVector: [], parsedValues: [] };
  }

  close(): void { this.base.close(); }

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

  private genRecovery(key: Buffer): string {
    const h = crypto.createHash('sha256').update(Buffer.concat([key, Buffer.from('recovery')])).digest();
    const indices = Array.from({length:12}, (_,i) => h.readUInt16BE(i*2) % WORDS.length);
    return indices.map(i => WORDS[i]).join(' ');
  }
}
