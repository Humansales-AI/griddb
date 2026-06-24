import {
  Token, ParserState, ParsedToken, ParsedNumber, ParsedWord,
  ParsedOperator, ChecksumResult, ParsedRecord,
} from './types';
import {
  NUMERIC_DIGIT_VALUE, WORD_CHAR, SPECIAL_CHAR, SPECIAL2_CHAR, OPERATOR_SYMBOL,
  DIGIT_TOKENS, NUMERIC_OPERATORS, NUMERIC_ANNOTATIONS, isDigitToken,
} from './tokens';

// SPECIAL3 command tokens (same values as Python)
import { Token as T } from './types';
const CMD_AUTH = T.D0;
const CMD_GRANT_R = T.D1;
const CMD_GRANT_W = T.D2;
const CMD_REVOKE = T.D3;
const CMD_ENCRYPT = T.D4;
const CMD_TOKENS = new Set([CMD_AUTH, CMD_GRANT_R, CMD_GRANT_W, CMD_REVOKE, CMD_ENCRYPT]);
const CMD_NAMES: Record<number, string> = {
  [CMD_AUTH]: 'AUTH', [CMD_GRANT_R]: 'GRANT_R', [CMD_GRANT_W]: 'GRANT_W',
  [CMD_REVOKE]: 'REVOKE', [CMD_ENCRYPT]: 'ENCRYPT',
};

export class Parser {
  state: ParserState = ParserState.NUM;
  accumulator: number[] = [];
  output: ParsedToken[] = [];
  records: ParsedRecord[] = [];
  currentRecordStart: number = 0;
  tokenCount: number = 0;

  reset(): void {
    this.state = ParserState.NUM;
    this.accumulator = [];
    this.output = [];
    this.records = [];
    this.currentRecordStart = 0;
    this.tokenCount = 0;
  }

  private finalizeNumber(): void {
    if (this.accumulator.length === 0) return;
    const digits = [...this.accumulator];
    const n = digits.length;
    // Use BigInt positional sum to avoid JS precision loss above 2^53
    let value: number | bigint = 0n;
    for (let i = 0; i < n; i++) {
      value += BigInt(digits[i]) * (10n ** BigInt(n - 1 - i));
    }
    // Convert to number if it fits safely, keep as bigint otherwise
    const asNum = Number(value);
    this.output.push({ type: 'number', digits, value: Number.isSafeInteger(asNum) ? asNum : value } as any);
    this.accumulator = [];
  }

  private finalizeWord(): void {
    const chars = this.accumulator.map(t => WORD_CHAR.get(t as Token) ?? '');
    this.output.push({ type: 'word', characters: chars, text: chars.join('') });
    this.accumulator = [];
  }

  private finalizeSpecial(): void {
    const chars = this.accumulator.map(t => SPECIAL_CHAR.get(t as Token) ?? '');
    this.output.push({ type: 'word', characters: chars, text: chars.join('') });
    this.accumulator = [];
  }

  private finalizeSpecial2(): void {
    const chars = this.accumulator.map(t => SPECIAL2_CHAR.get(t as Token) ?? '');
    this.output.push({ type: 'word', characters: chars, text: chars.join('') });
    this.accumulator = [];
  }

  private finalizeSpecial3(): void {
    if (this.accumulator.length === 0) return;
    const cmd = this.accumulator[0] as number;
    if (CMD_NAMES[cmd]) {
      this.output.push({ type: 'command', cmd: CMD_NAMES[cmd] } as any);
    }
    this.accumulator = [];
  }

  private emitRecord(): void {
    const recordTokens = this.output.slice(this.currentRecordStart);
    this.records.push({ tokens: [...recordTokens], bitOffset: this.currentRecordStart * 5 });
    this.output.push({ type: 'control', token: Token.RECORD });
    this.currentRecordStart = this.output.length;
  }

  feed(token: Token): ParsedToken | null {
    this.tokenCount++;
    let emitted: ParsedToken | null = null;

    if (this.state === ParserState.NUM) {
      if (token === Token.START) { this.finalizeNumber(); this.state = ParserState.WORD; }
      else if (token === Token.END) { this.finalizeNumber(); emitted = { type: 'control', token: Token.END }; }
      else if (token === Token.RECORD) { this.finalizeNumber(); this.emitRecord(); emitted = { type: 'control', token: Token.RECORD }; }
      else if (token === Token.CHECKSUM) { this.finalizeNumber(); emitted = { type: 'control', token: Token.CHECKSUM }; }
      else if (isDigitToken(token)) { const d = NUMERIC_DIGIT_VALUE.get(token); if (d != null) this.accumulator.push(d); }
      else if (NUMERIC_OPERATORS.has(token) || NUMERIC_ANNOTATIONS.has(token)) {
        this.finalizeNumber();
        const op: ParsedOperator = { type: 'operator', token, symbol: OPERATOR_SYMBOL.get(token) ?? '?' };
        this.output.push(op); emitted = op;
      }
      else throw new Error(`Unexpected token ${Token[token]} in NUM state`);

    } else if (this.state === ParserState.WORD) {
      if (token === Token.END) { this.finalizeWord(); this.state = ParserState.NUM; emitted = { type: 'control', token: Token.END }; }
      else if (token === Token.RECORD) { this.finalizeWord(); this.state = ParserState.NUM; this.emitRecord(); emitted = { type: 'control', token: Token.RECORD }; }
      else if (token === Token.START) { this.finalizeWord(); this.state = ParserState.SPECIAL; /* START-in-WORD → SPECIAL */ }
      else if (token === Token.CHECKSUM) { this.finalizeWord(); this.state = ParserState.NUM; emitted = { type: 'control', token: Token.CHECKSUM }; }
      else if (WORD_CHAR.has(token)) { this.accumulator.push(token as number); }
      else throw new Error(`Unexpected token ${Token[token]} in WORD state`);

    } else if (this.state === ParserState.SPECIAL) {
      if (token === Token.END) { this.finalizeSpecial(); this.state = ParserState.WORD; emitted = { type: 'control', token: Token.END }; }
      else if (token === Token.RECORD) { this.finalizeSpecial(); this.state = ParserState.NUM; this.emitRecord(); emitted = { type: 'control', token: Token.RECORD }; }
      else if (token === Token.CHECKSUM) { this.finalizeSpecial(); this.state = ParserState.NUM; emitted = { type: 'control', token: Token.CHECKSUM }; }
      else if (token === Token.START) { this.finalizeSpecial(); this.state = ParserState.SPECIAL2; /* START-in-SPECIAL → SPECIAL2 */ }
      else if (SPECIAL_CHAR.has(token)) { this.accumulator.push(token as number); }
      else throw new Error(`Unexpected token ${Token[token]} in SPECIAL state`);

    } else if (this.state === ParserState.SPECIAL2) {
      if (token === Token.END) { this.finalizeSpecial2(); this.state = ParserState.SPECIAL; emitted = { type: 'control', token: Token.END }; }
      else if (token === Token.RECORD) { this.finalizeSpecial2(); this.state = ParserState.NUM; this.emitRecord(); emitted = { type: 'control', token: Token.RECORD }; }
      else if (token === Token.CHECKSUM) { this.finalizeSpecial2(); this.state = ParserState.NUM; emitted = { type: 'control', token: Token.CHECKSUM }; }
      else if (token === Token.START) { this.finalizeSpecial2(); this.state = ParserState.SPECIAL3; /* START-in-SPECIAL2 → SPECIAL3 */ }
      else if (SPECIAL2_CHAR.has(token)) { this.accumulator.push(token as number); }
      else throw new Error(`Unexpected token ${Token[token]} in SPECIAL2 state`);

    } else if (this.state === ParserState.SPECIAL3) {
      if (token === Token.END) { this.finalizeSpecial3(); this.state = ParserState.SPECIAL2; emitted = { type: 'control', token: Token.END }; }
      else if (token === Token.RECORD) { this.finalizeSpecial3(); this.state = ParserState.NUM; this.emitRecord(); emitted = { type: 'control', token: Token.RECORD }; }
      else if (token === Token.CHECKSUM) { this.finalizeSpecial3(); this.state = ParserState.NUM; emitted = { type: 'control', token: Token.CHECKSUM }; }
      else if (token === Token.START) { /* deepest context */ }
      else if (CMD_TOKENS.has(token)) { this.accumulator = [token as number]; }
      else throw new Error(`Unexpected token ${Token[token]} in SPECIAL3 state`);
    }
    return emitted;
  }

  feedTokens(tokens: Token[]): ParsedToken[] {
    const emitted: ParsedToken[] = [];
    for (const t of tokens) { const r = this.feed(t); if (r !== null) emitted.push(r); }
    return emitted;
  }

  finalize(): void {
    if (this.state === ParserState.NUM) this.finalizeNumber();
    else if (this.state === ParserState.WORD) this.finalizeWord();
    else if (this.state === ParserState.SPECIAL) this.finalizeSpecial();
    else if (this.state === ParserState.SPECIAL2) this.finalizeSpecial2();
    else if (this.state === ParserState.SPECIAL3) this.finalizeSpecial3();
  }

  /** Reassemble fragmented words: merge consecutive WORD tokens, drop empties. */
  reassemble(): void {
    const merged: ParsedToken[] = [];
    let pending = '';
    for (const p of this.output) {
      if (p.type === 'word') {
        pending += (p as ParsedWord).text;
      } else {
        if (pending.length > 0) {
          merged.push({ type: 'word', characters: pending.split(''), text: pending });
          pending = '';
        }
        merged.push(p);
      }
    }
    if (pending.length > 0) {
      merged.push({ type: 'word', characters: pending.split(''), text: pending });
    }
    this.output = merged;
  }
}
