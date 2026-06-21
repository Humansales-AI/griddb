/**
 * Binary Grid Database — Parser
 * ==============================
 * Finite-state machine that reads 5-bit token streams.
 *
 * States: NUM (default), WORD
 * Transitions driven by START, END, RECORD, CHECKSUM control tokens.
 */

import {
  Token, ParserState, ParsedToken, ParsedNumber, ParsedWord,
  ParsedOperator, ChecksumResult, Record,
} from './types';
import {
  NUMERIC_DIGIT_VALUE, WORD_CHAR, OPERATOR_SYMBOL,
  DIGIT_TOKENS, NUMERIC_OPERATORS, NUMERIC_ANNOTATIONS, isDigitToken,
} from './tokens';

export class Parser {
  state: ParserState = ParserState.NUM;
  accumulator: number[] = [];
  output: ParsedToken[] = [];
  records: Record[] = [];
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

  /** Convert accumulated signed digits into a ParsedNumber. */
  private finalizeNumber(): void {
    if (this.accumulator.length === 0) return;
    const digits = [...this.accumulator];
    // Value = d₁*10^(n-1) + d₂*10^(n-2) + ... + dₙ*10^0
    const n = digits.length;
    let value = 0;
    for (let i = 0; i < n; i++) {
      value += digits[i] * Math.pow(10, n - 1 - i);
    }
    const parsed: ParsedNumber = { type: 'number', digits, value };
    this.output.push(parsed);
    this.accumulator = [];
  }

  /** Convert accumulated characters into a ParsedWord. Emits even empty words. */
  private finalizeWord(): void {
    const chars: string[] = [];
    for (const t of this.accumulator) {
      const ch = WORD_CHAR.get(t as Token);
      if (ch !== undefined) chars.push(ch);
    }
    const text = chars.join('');
    const parsed: ParsedWord = { type: 'word', characters: chars, text };
    this.output.push(parsed);
    this.accumulator = [];
  }

  /** Emit a RECORD boundary, grouping tokens since the last RECORD. */
  private emitRecord(): void {
    const recordTokens = this.output.slice(this.currentRecordStart);
    const record: Record = {
      tokens: [...recordTokens],
      bitOffset: this.currentRecordStart * 5,
    };
    this.records.push(record);
    this.output.push({ type: 'control', token: Token.RECORD });
    this.currentRecordStart = this.output.length;
  }

  /**
   * Feed a single 5-bit token into the parser.
   * Returns the parsed token if one was emitted, or null.
   */
  feed(token: Token): ParsedToken | null {
    this.tokenCount++;
    let emitted: ParsedToken | null = null;

    if (this.state === ParserState.NUM) {
      if (token === Token.START) {
        this.finalizeNumber();
        this.state = ParserState.WORD;
      } else if (token === Token.END) {
        this.finalizeNumber();
        emitted = { type: 'control', token: Token.END };
      } else if (token === Token.RECORD) {
        this.finalizeNumber();
        this.emitRecord();
        emitted = { type: 'control', token: Token.RECORD };
      } else if (token === Token.CHECKSUM) {
        this.finalizeNumber();
        emitted = { type: 'control', token: Token.CHECKSUM };
      } else if (isDigitToken(token)) {
        const d = NUMERIC_DIGIT_VALUE.get(token);
        if (d !== null && d !== undefined) {
          this.accumulator.push(d);
        }
      } else if (NUMERIC_OPERATORS.has(token)) {
        this.finalizeNumber();
        const op: ParsedOperator = {
          type: 'operator',
          token,
          symbol: OPERATOR_SYMBOL.get(token) ?? '?',
        };
        this.output.push(op);
        emitted = op;
      } else if (NUMERIC_ANNOTATIONS.has(token)) {
        // Storage annotations (S for Scale) — not arithmetic operators.
        this.finalizeNumber();
        const op: ParsedOperator = {
          type: 'operator',
          token,
          symbol: OPERATOR_SYMBOL.get(token) ?? '?',
        };
        this.output.push(op);
        emitted = op;
      } else {
        throw new Error(`Unexpected token ${Token[token]} in NUM state`);
      }
    } else if (this.state === ParserState.WORD) {
      if (token === Token.END) {
        this.finalizeWord();
        this.state = ParserState.NUM;
        emitted = { type: 'control', token: Token.END };
      } else if (token === Token.RECORD) {
        this.finalizeWord();
        this.state = ParserState.NUM;
        this.emitRecord();
        emitted = { type: 'control', token: Token.RECORD };
      } else if (token === Token.START) {
        throw new Error('Nested START not allowed in WORD context');
      } else if (token === Token.CHECKSUM) {
        this.finalizeWord();
        this.state = ParserState.NUM;
        emitted = { type: 'control', token: Token.CHECKSUM };
      } else if (WORD_CHAR.has(token)) {
        this.accumulator.push(token as number);
      } else {
        throw new Error(`Unexpected token ${Token[token]} in WORD state`);
      }
    }
    return emitted;
  }

  /** Feed a list of tokens. Returns all emitted parsed tokens. */
  feedTokens(tokens: Token[]): ParsedToken[] {
    const emitted: ParsedToken[] = [];
    for (const t of tokens) {
      const result = this.feed(t);
      if (result !== null) emitted.push(result);
    }
    return emitted;
  }

  /** Finalize any pending accumulator. */
  finalize(): void {
    if (this.state === ParserState.NUM) {
      this.finalizeNumber();
    } else if (this.state === ParserState.WORD) {
      this.finalizeWord();
    }
  }
}
