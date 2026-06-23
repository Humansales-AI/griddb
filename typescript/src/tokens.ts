/**
 * Binary Grid Database — Token Mappings
 * ======================================
 * All bidirectional lookup tables for the 32-slot vocabulary.
 */

import { Token } from './types';

// ── Numeric context: token → digit value ──────────────────────────────────

export const NUMERIC_DIGIT_VALUE: Map<Token, number | null> = new Map([
  [Token.D0, 0],  [Token.D1, 1],  [Token.D2, 2],  [Token.D3, 3],  [Token.D4, 4],
  [Token.D5, 5],  [Token.D6, 6],  [Token.D7, 7],  [Token.D8, 8],  [Token.D9, 9],
  [Token.N1, -1], [Token.N2, -2], [Token.N3, -3], [Token.N4, -4], [Token.N5, -5],
  [Token.N6, -6], [Token.N7, -7], [Token.N8, -8], [Token.N9, -9],
]);

// Numeric context: digit value → token
export const DIGIT_TO_TOKEN: Map<number, Token> = new Map();
for (const [tok, val] of NUMERIC_DIGIT_VALUE) {
  if (val !== null) DIGIT_TO_TOKEN.set(val, tok);
}

// Set of tokens representing digits
export const DIGIT_TOKENS: Set<Token> = new Set(NUMERIC_DIGIT_VALUE.keys());

// ── Numeric context: arithmetic operators ──────────────────────────────────

export const NUMERIC_OPERATORS: Set<Token> = new Set([
  Token.T_PLUS, Token.T_MINUS, Token.T_MUL, Token.T_DIV,
  Token.T_EQ, Token.T_LPAREN, Token.T_RPAREN, Token.T_POW,
]);

// ── Numeric context: storage annotations (S = Scale) ───────────────────────
// S is NOT an arithmetic operator — it's metadata telling the application
// layer "this integer has N implied decimal places." The database stores
// pure integers; decimal arithmetic happens above.

export const NUMERIC_ANNOTATIONS: Set<Token> = new Set([Token.T_SCALE]);

// ── Operator symbol mappings ───────────────────────────────────────────────

export const OPERATOR_SYMBOL: Map<Token, string> = new Map([
  [Token.T_PLUS, '+'], [Token.T_MINUS, '-'], [Token.T_MUL, '*'], [Token.T_DIV, '/'],
  [Token.T_EQ, '='], [Token.T_LPAREN, '('], [Token.T_RPAREN, ')'],
  [Token.T_POW, '^'], [Token.T_SCALE, 'S'],
]);

export const SYMBOL_TO_OPERATOR: Map<string, Token> = new Map();
for (const [tok, sym] of OPERATOR_SYMBOL) {
  SYMBOL_TO_OPERATOR.set(sym, tok);
}

// ── Word context: token → character ────────────────────────────────────────

export const WORD_CHAR: Map<Token, string> = new Map([
  [Token.D0, 'A'], [Token.D1, 'B'], [Token.D2, 'C'], [Token.D3, 'D'],
  [Token.D4, 'E'], [Token.D5, 'F'], [Token.D6, 'G'], [Token.D7, 'H'],
  [Token.D8, 'I'], [Token.D9, 'J'],
  [Token.T_PLUS, 'K'], [Token.T_MINUS, 'L'], [Token.T_MUL, 'M'], [Token.T_DIV, 'N'],
  [Token.T_EQ, 'O'], [Token.T_LPAREN, 'P'], [Token.T_RPAREN, 'Q'],
  [Token.N1, 'R'], [Token.N2, 'S'], [Token.N3, 'T'], [Token.N4, 'U'],
  [Token.N5, 'V'], [Token.N6, 'W'], [Token.N7, 'X'], [Token.N8, 'Y'], [Token.N9, 'Z'],
  [Token.T_POW, ' '],    // SPACE
  [Token.T_SCALE, '.'],  // PERIOD
]);

export const CHAR_TO_WORD_TOKEN: Map<string, Token> = new Map();
for (const [tok, ch] of WORD_CHAR) {
  CHAR_TO_WORD_TOKEN.set(ch, tok);
}

// ── SPECIAL context: lowercase letters & special characters ──────────────
// Triggered by START-in-WORD. Uses same 28 slots (00000-11011) but remapped.
// Controls (11100-11111) retain their meaning across all contexts.

export const SPECIAL_CHAR: Map<Token, string> = new Map([
  [Token.D0, 'a'], [Token.D1, 'b'], [Token.D2, 'c'], [Token.D3, 'd'], [Token.D4, 'e'],
  [Token.D5, 'f'], [Token.D6, 'g'], [Token.D7, 'h'], [Token.D8, 'i'], [Token.D9, 'j'],
  [Token.T_PLUS, 'k'], [Token.T_MINUS, 'l'], [Token.T_MUL, 'm'], [Token.T_DIV, 'n'],
  [Token.T_EQ, 'o'], [Token.T_LPAREN, 'p'], [Token.T_RPAREN, 'q'],
  [Token.N1, 'r'], [Token.N2, 's'], [Token.N3, 't'], [Token.N4, 'u'],
  [Token.N5, 'v'], [Token.N6, 'w'], [Token.N7, 'x'], [Token.N8, 'y'], [Token.N9, 'z'],
  [Token.T_POW, '@'],      // was SPACE in WORD
  [Token.T_SCALE, '-'],     // was . in WORD
]);

export const CHAR_TO_SPECIAL_TOKEN: Map<string, Token> = new Map();
for (const [tok, ch] of SPECIAL_CHAR) {
  CHAR_TO_SPECIAL_TOKEN.set(ch, tok);
}

// ── SPECIAL2 context: extended punctuation ──────────────────────────────

export const SPECIAL2_CHAR: Map<Token, string> = new Map([
  [Token.D0, '!'], [Token.D1, '"'], [Token.D2, '#'], [Token.D3, '$'],
  [Token.D4, '%'], [Token.D5, '&'], [Token.D6, "'"], [Token.D7, '('],
  [Token.D8, ')'], [Token.D9, '*'],
  [Token.T_PLUS, '+'], [Token.T_MINUS, ','], [Token.T_MUL, '/'],
  [Token.T_DIV, ':'], [Token.T_EQ, ';'], [Token.T_LPAREN, '<'],
  [Token.T_RPAREN, '='], [Token.N1, '>'], [Token.N2, '?'],
  [Token.N3, '['], [Token.N4, '\\'], [Token.N5, ']'],
  [Token.N6, '^'], [Token.N7, '_'], [Token.N8, '`'],
  [Token.N9, '{'], [Token.T_POW, '|'], [Token.T_SCALE, '}'],
]);
export const CHAR_TO_SPECIAL2_TOKEN: Map<string, Token> = new Map(
  [...SPECIAL2_CHAR].map(([k, v]) => [v, k])
);

// ── Control tokens ─────────────────────────────────────────────────────────

export const CONTROL_TOKENS: Set<Token> = new Set([
  Token.START, Token.END, Token.RECORD, Token.CHECKSUM,
]);

/** Check if a token is a digit token. */
export function isDigitToken(tok: Token): boolean {
  return DIGIT_TOKENS.has(tok);
}

/** Check if a token is a control token. */
export function isControlToken(tok: Token): boolean {
  return CONTROL_TOKENS.has(tok);
}

/** Check if a token is an arithmetic operator. */
export function isOperator(tok: Token): boolean {
  return NUMERIC_OPERATORS.has(tok);
}

/** Check if a token is a storage annotation (e.g., S for Scale). */
export function isAnnotation(tok: Token): boolean {
  return NUMERIC_ANNOTATIONS.has(tok);
}
