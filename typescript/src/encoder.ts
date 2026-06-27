/**
 * Binary Grid Database — Encoder
 * ===============================
 * Converts high-level values (integers, words, expressions, records)
 * into 5-bit Token streams.
 */

import { Token } from './types';
import { DIGIT_TO_TOKEN, SYMBOL_TO_OPERATOR, CHAR_TO_WORD_TOKEN, CHAR_TO_SPECIAL_TOKEN, CHAR_TO_SPECIAL2_TOKEN } from './tokens';

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
  static encodeInteger(value: number | bigint): Token[] {
    if (value === 0 || value === 0n) return [Token.D0, Token.END];

    const sign = typeof value === 'bigint' ? (value < 0n ? -1 : 1) : (value >= 0 ? 1 : -1);
    // Use BigInt for all values to avoid JS number precision loss above 2^53
    const n = typeof value === 'bigint' ? value : BigInt(value);
    const abs = n < 0n ? -n : n;
    const digitsStr = abs.toString();

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
   * Encode a word as START + letters + END.
   * Handles mixed WORD/SPECIAL contexts via START-in-WORD switching.
   *
   * Example: "HI" → [START, H, I, END] = [11111, 00111, 01000, 11110]
   * Example: "hi" → [START, START, h, i, END, END]
   * Example: "hi@there" → START START h i @ t h e r e END END
   */
  static encodeWord(text: string): Token[] {
    const tokens: Token[] = [Token.START];
    let depth = 0; // 0=WORD, 1=SPECIAL, 2=SPECIAL2
    const pop = (target: number) => { for (let i = depth; i > target; i--) tokens.push(Token.END); depth = target; };

    for (const ch of text) {
      if (ch >= '0' && ch <= '9') {
        pop(0); tokens.push(Token.END); tokens.push(DIGIT_TO_TOKEN.get(parseInt(ch, 10))!); tokens.push(Token.START);
        continue;
      }
      const wordTok = CHAR_TO_WORD_TOKEN.get(ch);
      if (wordTok !== undefined) { pop(0); tokens.push(wordTok); continue; }

      const specialTok = CHAR_TO_SPECIAL_TOKEN.get(ch);
      if (specialTok !== undefined) { if (depth > 1) pop(1); else if (depth < 1) { tokens.push(Token.START); depth = 1; } tokens.push(specialTok); continue; }

      const special2Tok = CHAR_TO_SPECIAL2_TOKEN.get(ch);
      if (special2Tok !== undefined) { if (depth < 2) { if (depth < 1) { tokens.push(Token.START); depth = 1; } tokens.push(Token.START); depth = 2; } tokens.push(special2Tok); continue; }

      const upperTok = CHAR_TO_WORD_TOKEN.get(ch.toUpperCase());
      if (upperTok !== undefined) { pop(0); tokens.push(upperTok); continue; }

      throw new Error(`Character '${ch}' cannot be encoded in any context`);
    }
    pop(0); tokens.push(Token.END);
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

  /** Label a cell position with metadata text. Position in NUM context. */
  static encodeLabel(position: number, label: string): Token[] {
    const CMD_LABEL = Token.D5;
    return [
      Token.START, Token.START, Token.START, Token.START,  // enter SPECIAL3
      CMD_LABEL,                                             // the LABEL command
      Token.END, Token.END, Token.END, Token.END,            // pop SPECIAL3→SPECIAL2→SPECIAL→WORD
      Token.END,                                             // pop WORD→NUM
      ...Encoder.encodeInteger(position),                    // position in NUM context ✓
      Token.START,                                           // re-enter WORD
      ...Encoder.encodeWord(label),                          // label text
      Token.END,                                             // WORD→NUM
    ];
  }
}
