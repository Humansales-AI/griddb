/**
 * Binary Grid Database — Encoder
 * ===============================
 * Converts high-level values (integers, words, expressions, records)
 * into 5-bit Token streams.
 */

import { Token } from './types';
import { DIGIT_TO_TOKEN, SYMBOL_TO_OPERATOR, CHAR_TO_WORD_TOKEN } from './tokens';

export class Encoder {
  /**
   * Encode a signed integer as signed-digit tokens + END.
   *
   * Example: 123  → [D1, D2, D3, END]
   * Example: -123 → [N1, N2, N3, END]
   * Example: 0    → [D0, END]
   *
   * For negative numbers, each digit carries its own sign.
   * Zero digits in negative numbers use D0 (no -0 concept).
   */
  static encodeInteger(value: number): Token[] {
    if (value === 0) return [Token.D0, Token.END];

    const sign = value >= 0 ? 1 : -1;
    const digitsStr = Math.abs(value).toString();

    const tokens: Token[] = [];
    for (const ch of digitsStr) {
      const d = parseInt(ch, 10);
      if (d === 0) {
        tokens.push(Token.D0);
      } else {
        const prefix = sign < 0 ? 'N' : 'D';
        const tokenName = `${prefix}${d}` as keyof typeof Token;
        tokens.push(Token[tokenName]);
      }
    }
    tokens.push(Token.END);
    return tokens;
  }

  /** Encode a list of signed digit values (without trailing END). */
  static encodeSignedDigits(digits: number[]): Token[] {
    return digits.map(d => {
      const tok = DIGIT_TO_TOKEN.get(d);
      if (tok === undefined) throw new Error(`Invalid digit value: ${d}`);
      return tok;
    });
  }

  /** Encode signed digits + END. */
  static encodeNumberFromDigits(digits: number[]): Token[] {
    return [...Encoder.encodeSignedDigits(digits), Token.END];
  }

  /**
   * Encode a word as START + letter tokens + END.
   *
   * Example: "HI" → [START, H, I, END] = [11111, 00111, 01000, 11110]
   */
  static encodeWord(text: string): Token[] {
    const tokens: Token[] = [Token.START];
    for (const ch of text.toUpperCase()) {
      const tok = CHAR_TO_WORD_TOKEN.get(ch);
      if (tok === undefined) throw new Error(`Character '${ch}' cannot be encoded`);
      tokens.push(tok);
    }
    tokens.push(Token.END);
    return tokens;
  }

  /** Encode an operator symbol to its token. */
  static encodeOperator(symbol: string): Token {
    const tok = SYMBOL_TO_OPERATOR.get(symbol);
    if (tok === undefined) throw new Error(`Unknown operator: '${symbol}'`);
    return tok;
  }

  /**
   * Encode an expression from a list of values, operators, and digit-lists.
   *
   * Each element can be:
   *   - number: a complete integer (e.g., 123, -5)
   *   - string: an operator (e.g., '+', '*', 'S')
   *   - number[]: signed digits forming a multi-digit number (e.g., [-1, -2, -3])
   */
  static encodeExpression(items: (number | string | number[])[]): Token[] {
    const result: Token[] = [];
    for (const item of items) {
      if (typeof item === 'number') {
        result.push(...Encoder.encodeInteger(item));
      } else if (Array.isArray(item)) {
        result.push(...Encoder.encodeNumberFromDigits(item));
      } else if (typeof item === 'string') {
        result.push(Encoder.encodeOperator(item));
      }
    }
    return result;
  }

  /**
   * Encode values into a record terminated by RECORD.
   *
   * Each value can be a number (integer), a string (word), or a pre-encoded
   * token array.  The RECORD marker delimits logical tuples for geometric queries.
   */
  static encodeRecord(...values: (number | string | Token[])[]): Token[] {
    const tokens: Token[] = [];
    for (const val of values) {
      if (typeof val === 'number') {
        tokens.push(...Encoder.encodeInteger(val));
      } else if (typeof val === 'string') {
        tokens.push(...Encoder.encodeWord(val));
      } else if (Array.isArray(val)) {
        tokens.push(...val);
      }
    }
    tokens.push(Token.RECORD);
    return tokens;
  }
}
