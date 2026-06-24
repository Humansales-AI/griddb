/** 5bit CryptoRLS — encrypted storage with per-user keys. TypeScript. */
import crypto from 'crypto';
import { AllocGrid, Encoder, Token, Parser, ParsedNumber, packToBytes, unpackFromBytes, AllocRecord } from '../index';
import { PermissionDenied } from './engine';

const keyCache = new Map<string, Buffer>();

function deriveKey(uid: number, secret: Buffer): Buffer {
  const k = `${uid}:${secret.toString('hex')}`;
  if (!keyCache.has(k)) keyCache.set(k, crypto.pbkdf2Sync(secret, String(uid), 100_000, 32, 'sha256'));
  return keyCache.get(k)!;
}

function ctrEncrypt(key: Buffer, pt: Buffer): Buffer {
  const nonce = crypto.randomBytes(16); const r = Buffer.concat([nonce, Buffer.alloc(pt.length)]);
  for (let i = 0; i < pt.length; i += 32) {
    const ctr = Buffer.concat([nonce, Buffer.from([0,0,0,0,0,0,0,0])]); ctr.writeBigUInt64BE(BigInt(i/32), 16);
    const ks = crypto.createHmac('sha256', key).update(ctr).digest();
    for (let j = i; j < Math.min(i+32, pt.length); j++) r[16+j] = pt[j] ^ ks[j-i];
  }
  return r;
}

function ctrDecrypt(key: Buffer, ct: Buffer): Buffer {
  const nonce = ct.subarray(0, 16); const data = ct.subarray(16);
  const r = Buffer.alloc(data.length);
  for (let i = 0; i < data.length; i += 32) {
    const ctr = Buffer.concat([nonce, Buffer.from([0,0,0,0,0,0,0,0])]); ctr.writeBigUInt64BE(BigInt(i/32), 16);
    const ks = crypto.createHmac('sha256', key).update(ctr).digest();
    for (let j = i; j < Math.min(i+32, data.length); j++) r[j] = data[j] ^ ks[j-i];
  }
  return r;
}

export class CryptoRLS extends AllocGrid {
  private bypass: number | null = null;
  constructor(dataDir: string) { super(dataDir); }

  write(rid: number, uid: number, secret: Buffer, tokens: Token[]): number {
    const key = deriveKey(uid, secret);
    const [packed, pad] = packToBytes(tokens);
    const encrypted = ctrEncrypt(key, Buffer.from(packed));
    const mac = crypto.createHmac('sha256', key).update(encrypted).digest().subarray(0, 16);
    const blob = Buffer.concat([Buffer.from([pad]), encrypted, mac]);
    const bt: Token[] = []; for (const b of blob) bt.push(...Encoder.encodeInteger(b));
    bt.push(Token.RECORD); return super.write(rid, bt);
  }

  read(rid: number, uid: number, secret: Buffer): AllocRecord | null {
    const rec = super.read(rid); if (!rec || rec.isTombstone) return null;
    const nums = rec.parsed.filter(p => p.type === 'number').map((p: any) => p.value);
    if (nums.length < 18) throw new PermissionDenied('Corrupted');
    const blob = Buffer.from(nums), pad = blob[0], encrypted = blob.subarray(1, blob.length-16), mac = blob.subarray(blob.length-16);
    const key = deriveKey(uid, secret);
    if (!crypto.timingSafeEqual(mac, crypto.createHmac('sha256', key).update(encrypted).digest().subarray(0,16))) throw new PermissionDenied('Wrong key');
    const pt = ctrDecrypt(key, encrypted);
    const tokens = unpackFromBytes(new Uint8Array(pt), pad);
    const p = new Parser(); p.feedTokens(tokens); p.finalize();
    if (typeof (p as any).reassemble === 'function') (p as any).reassemble();
    return { recordId: rid, tokens, parsed: p.output, byteOffset: rec.byteOffset,
             bitLength: tokens.length*5, isTombstone: false, valueVector: [], digitVector: [], parsedValues: [] };
  }

  delete(rid: number, uid: number, secret: Buffer): boolean { this.read(rid, uid, secret); return super.delete(rid); }
  asAdmin(): CryptoRLS { this.bypass = -1; return this; }
}
