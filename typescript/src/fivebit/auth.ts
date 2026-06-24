/**
 * 5bit Auth — TypeScript User & Session Management
 * ==================================================
 * Drop-in auth on AllocGrid. PBKDF2 hashing, session tokens.
 */
import crypto from 'crypto';
import { AllocGrid, Encoder, Token, Parser, ParsedNumber, ParsedWord } from '../index';

const SESSION_BASE = 10_000_000;
const FIELD_STRIDE = 100;

export class AuthStore {
  private grid: AllocGrid;

  constructor(dataDir: string) { this.grid = new AllocGrid(dataDir); }

  private hash(pw: string): string {
    const salt = crypto.randomBytes(16).toString('hex');
    const h = crypto.pbkdf2Sync(pw, salt, 100_000, 32, 'sha256').toString('hex');
    return salt + '$' + h;
  }
  private verify(pw: string, stored: string): boolean {
    try {
      const [s, h] = stored.split('$');
      return h === crypto.pbkdf2Sync(pw, s, 100_000, 32, 'sha256').toString('hex');
    } catch { return false; }
  }
  private rid(uid: number, field: number): number { return uid * FIELD_STRIDE + field; }

  private readField(uid: number, field: number): string {
    const rec = this.grid.read(this.rid(uid, field));
    if (!rec) return '';
    return rec.parsed.filter(p => p.type === 'word').map((p: any) => p.text).join('');
  }
  private writeField(uid: number, field: number, value: string): void {
    this.grid.write(this.rid(uid, field), [
      ...Encoder.encodeInteger(uid), ...Encoder.encodeInteger(field),
      ...Encoder.encodeWord(value), Token.RECORD,
    ]);
  }

  signup(email: string, password: string, name: string = ''): number {
    let uid = 1;
    while (this.readField(uid, 1)) uid++;
    this.writeField(uid, 1, email);
    this.writeField(uid, 2, this.hash(password));
    this.writeField(uid, 3, name);
    return uid;
  }

  login(email: string, password: string): { token: string; userId: number } | null {
    for (let uid = 1; uid < 1000; uid++) {
      const e = this.readField(uid, 1);
      if (!e) break;
      if (e === email) {
        const h = this.readField(uid, 2);
        if (this.verify(password, h)) return this.createSession(uid);
      }
    }
    return null;
  }

  getUser(uid: number) {
    const email = this.readField(uid, 1);
    if (!email) return null;
    return { id: uid, email, hash: this.readField(uid, 2), name: this.readField(uid, 3) };
  }

  private createSession(uid: number) {
    const token = crypto.randomBytes(32).toString('hex');
    const rid = SESSION_BASE + (this._hashToken(token) & 0xFFFFF);
    this.grid.write(rid, [
      ...Encoder.encodeWord(token), ...Encoder.encodeInteger(uid),
      ...Encoder.encodeInteger(Math.floor(Date.now()/1000 + 86400)), Token.RECORD,
    ]);
    return { token, userId: uid };
  }

  verifySession(token: string): number | null {
    const rid = SESSION_BASE + (this._hashToken(token) & 0xFFFFF);
    const rec = this.grid.read(rid);
    if (!rec) return null;
    const words = rec.parsed.filter(p => p.type === 'word').map((p: any) => p.text);
    const nums = rec.parsed.filter(p => p.type === 'number').map((p: any) => p.value);
    if (words[0] === token && nums.length >= 2 && Date.now()/1000 < nums[1]) return nums[0];
    return null;
  }

  logout(token: string): void {
    this.grid.delete(SESSION_BASE + (this._hashToken(token) & 0xFFFFF));
  }

  private _hashToken(t: string): number {
    let h = 0;
    for (let i = 0; i < t.length; i++) h = ((h << 5) - h) + t.charCodeAt(i) | 0;
    return Math.abs(h);
  }

  close(): void { this.grid.close(); }
}
