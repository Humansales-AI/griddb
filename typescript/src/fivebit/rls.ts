/**
 * 5bit RLS — TypeScript Row-Level Security
 * ==========================================
 * RLSEngine extends AllocGrid (not a wrapper). Owner at position 0.
 * CryptoRLS encrypts records with per-user AES key.
 */
import crypto from 'crypto';
import { AllocGrid, Encoder, Token, Parser, ParsedNumber, AllocRecord } from '../index';

export class PermissionDenied extends Error {
  constructor(msg: string) { super(msg); this.name = 'PermissionDenied'; }
}

// ── RLSEngine ────────────────────────────────────────────────────────────

export class RLSEngine extends AllocGrid {
  private ownerPosition: number;
  private bypassUser: number | null = null;

  constructor(dataDir: string, ownerPosition: number = 0) {
    super(dataDir);
    this.ownerPosition = ownerPosition;
  }

  private getOwner(recordId: number): number | null {
    const rec = super.read(recordId);
    if (!rec || rec.isTombstone) return null;
    const vals = rec.parsed.filter(p => p.type === 'number').map((p: any) => p.value);
    return vals.length > this.ownerPosition ? vals[this.ownerPosition] : (vals[0] || null);
  }

  read(recordId: number, userId: number): AllocRecord | null {
    const owner = this.getOwner(recordId);
    if (owner !== null && owner !== userId && this.bypassUser === null) {
      throw new PermissionDenied(`RLS: user ${userId} cannot read record ${recordId}`);
    }
    return super.read(recordId);
  }

  write(recordId: number, userId: number, tokens: Token[]): number {
    const existing = this.getOwner(recordId);
    if (existing !== null && existing !== userId && this.bypassUser === null) {
      throw new PermissionDenied(`RLS: user ${userId} cannot write record ${recordId}`);
    }
    // Prepend owner token at position 0
    const ownerTokens = Encoder.encodeInteger(userId);
    return super.write(recordId, [...ownerTokens, ...tokens]);
  }

  delete(recordId: number, userId: number): boolean {
    const owner = this.getOwner(recordId);
    if (owner !== null && owner !== userId && this.bypassUser === null) {
      throw new PermissionDenied(`RLS: user ${userId} cannot delete record ${recordId}`);
    }
    return super.delete(recordId);
  }

  asAdmin(): RLSEngine { this.bypassUser = -1; return this; }
  clearAdmin(): void { this.bypassUser = null; }
}

// ── CryptoRLS ────────────────────────────────────────────────────────────

function deriveKey(userId: number, secret: Buffer): Buffer {
  return crypto.pbkdf2Sync(secret, String(userId), 100_000, 32, 'sha256');
}

export class CryptoRLS extends AllocGrid {
  private bypassUser: number | null = null;

  constructor(dataDir: string) { super(dataDir); }

  write(recordId: number, userId: number, secretKey: Buffer, tokens: Token[]): number {
    // Pack tokens
    const { packToBytes } = require('../serialization');
    const [packed, pad] = packToBytes(tokens);
    const plaintext = Buffer.from(packed);

    // Encrypt with CTR + HMAC
    const key = deriveKey(userId, secretKey);
    const nonce = crypto.randomBytes(16);
    const ciphertext = this._ctrEncrypt(key, nonce, plaintext);
    const mac = crypto.createHmac('sha256', key).update(ciphertext).digest().subarray(0, 16);
    const blob = Buffer.concat([Buffer.from([pad]), nonce, ciphertext, mac]);

    // Encode as NUM tokens
    const blobTokens: Token[] = [];
    for (const b of blob) blobTokens.push(...Encoder.encodeInteger(b));
    blobTokens.push(Token.RECORD);
    return super.write(recordId, blobTokens);
  }

  read(recordId: number, userId: number, secretKey: Buffer): AllocRecord | null {
    const rec = super.read(recordId);
    if (!rec || rec.isTombstone) return null;
    const nums = rec.parsed.filter(p => p.type === 'number').map((p: any) => p.value);
    if (nums.length < 18) throw new PermissionDenied('Corrupted record');
    const blob = Buffer.from(nums);
    const pad = blob[0];
    const nonce = blob.subarray(1, 17);
    const mac = blob.subarray(blob.length - 16);
    const ciphertext = blob.subarray(17, blob.length - 16);

    const key = deriveKey(userId, secretKey);
    const expectedMac = crypto.createHmac('sha256', key).update(ciphertext).digest().subarray(0, 16);
    if (!crypto.timingSafeEqual(mac, expectedMac)) {
      throw new PermissionDenied('Wrong key or corrupted data');
    }
    const plaintext = this._ctrDecrypt(key, nonce, ciphertext);
    const { unpackFromBytes } = require('../serialization');
    const tokens = unpackFromBytes(new Uint8Array(plaintext), pad);
    const parser = new Parser();
    parser.feedTokens(tokens); parser.finalize();
    if (typeof (parser as any).reassemble === 'function') (parser as any).reassemble();
    return { recordId, tokens, parsed: parser.output, byteOffset: rec.byteOffset,
             bitLength: tokens.length * 5, isTombstone: false,
             valueVector: [], digitVector: [], parsedValues: [] };
  }

  delete(recordId: number, userId: number, secretKey: Buffer): boolean {
    this.read(recordId, userId, secretKey); // Verify ownership
    return super.delete(recordId);
  }

  asAdmin(): CryptoRLS { this.bypassUser = -1; return this; }

  private _ctrEncrypt(key: Buffer, nonce: Buffer, data: Buffer): Buffer {
    const result = Buffer.alloc(data.length);
    for (let i = 0; i < data.length; i += 32) {
      const ctr = Buffer.concat([nonce, Buffer.from([0,0,0,0,0,0,0,0])]);
      ctr.writeBigUInt64BE(BigInt(i/32), 16);
      const ks = crypto.createHash('sha256').update(Buffer.concat([key, ctr])).digest();
      const end = Math.min(i + 32, data.length);
      for (let j = i; j < end; j++) result[j] = data[j] ^ ks[j - i];
    }
    return result;
  }
  private _ctrDecrypt = this._ctrEncrypt; // CTR is symmetric
}
