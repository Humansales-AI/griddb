/**
 * Binary Grid Database — Core Types
 * ==================================
 * Shared types and interfaces for the 5-bit binary token system.
 */

// ── Token enum (all 32 five-bit codes) ─────────────────────────────────────

export enum Token {
  // Positive digits (0x00–0x09)
  D0 = 0b00000,
  D1 = 0b00001,
  D2 = 0b00010,
  D3 = 0b00011,
  D4 = 0b00100,
  D5 = 0b00101,
  D6 = 0b00110,
  D7 = 0b00111,
  D8 = 0b01000,
  D9 = 0b01001,

  // Operators / Letters (0x0A–0x14)
  T_PLUS   = 0b01010,  // + in NUM, K in WORD
  T_MINUS  = 0b01011,  // - in NUM, L in WORD
  T_MUL    = 0b01100,  // * in NUM, M in WORD
  T_DIV    = 0b01101,  // / in NUM, N in WORD
  T_EQ     = 0b01110,  // = in NUM, O in WORD
  T_LPAREN = 0b01111,  // ( in NUM, P in WORD
  T_RPAREN = 0b10000,  // ) in NUM, Q in WORD

  // Negative digits (0x11–0x19)
  N1 = 0b10001,  // -1 in NUM, R in WORD
  N2 = 0b10010,  // -2 in NUM, S in WORD
  N3 = 0b10011,  // -3 in NUM, T in WORD
  N4 = 0b10100,  // -4 in NUM, U in WORD
  N5 = 0b10101,  // -5 in NUM, V in WORD
  N6 = 0b10110,  // -6 in NUM, W in WORD
  N7 = 0b10111,  // -7 in NUM, X in WORD
  N8 = 0b11000,  // -8 in NUM, Y in WORD
  N9 = 0b11001,  // -9 in NUM, Z in WORD

  // Extended operators / punctuation (0x1A–0x1B)
  T_POW   = 0b11010,  // ^ in NUM, SPACE in WORD
  T_SCALE = 0b11011,  // S in NUM, . in WORD

  // Control codes (0x1C–0x1F)
  RECORD   = 0b11100,
  CHECKSUM = 0b11101,
  END      = 0b11110,
  START    = 0b11111,
}

// ── Parser state ────────────────────────────────────────────────────────────

export enum ParserState {
  NUM = 0,
  WORD = 1,
  SPECIAL = 2,
  SPECIAL2 = 3,
}

// ── Parsed token types ──────────────────────────────────────────────────────

export interface ParsedNumber {
  type: 'number';
  digits: number[];   // Signed digit values, e.g., [-1, -2, -3]
  value: number;       // Computed integer value
}

export interface ParsedWord {
  type: 'word';
  characters: string[];
  text: string;
}

export interface ParsedOperator {
  type: 'operator';
  token: Token;
  symbol: string;
}

export interface ParsedScaledNumber {
  type: 'scaled';
  numerator: number;
  scale: number;       // Number of decimal places (non-negative)
  asFloat: number;     // numerator / 10^scale
}

export interface ChecksumResult {
  type: 'checksum';
  expected: number;
  computed: number;
  passed: boolean;
}

export type ParsedToken =
  | ParsedNumber
  | ParsedWord
  | ParsedOperator
  | ParsedScaledNumber
  | ChecksumResult
  | { type: 'control'; token: Token };

// ── Parsed Record ──────────────────────────────────────────────────────────

export interface ParsedRecord {
  tokens: ParsedToken[];
  bitOffset: number;
}

// ── Grid record ─────────────────────────────────────────────────────────────

export interface GridRecord {
  tokens: Token[];
  bitOffset: number;
  bitLength: number;
  parsedValues: ParsedNumber[];
  valueVector: number[];
  digitVector: number[];
}

// ── Token display names ─────────────────────────────────────────────────────

export const TOKEN_NAME: Record<Token, string> = {
  [Token.D0]: '0', [Token.D1]: '1', [Token.D2]: '2', [Token.D3]: '3', [Token.D4]: '4',
  [Token.D5]: '5', [Token.D6]: '6', [Token.D7]: '7', [Token.D8]: '8', [Token.D9]: '9',
  [Token.N1]: '-1', [Token.N2]: '-2', [Token.N3]: '-3', [Token.N4]: '-4', [Token.N5]: '-5',
  [Token.N6]: '-6', [Token.N7]: '-7', [Token.N8]: '-8', [Token.N9]: '-9',
  [Token.T_PLUS]: '+', [Token.T_MINUS]: '-', [Token.T_MUL]: '*', [Token.T_DIV]: '/',
  [Token.T_EQ]: '=', [Token.T_LPAREN]: '(', [Token.T_RPAREN]: ')',
  [Token.T_POW]: '^', [Token.T_SCALE]: 'S',
  [Token.RECORD]: 'RECORD', [Token.CHECKSUM]: 'CHECKSUM',
  [Token.END]: 'END', [Token.START]: 'START',
};

/**
 * Format a list of tokens as a space-separated binary string (for debugging).
 */
export function tokenStreamToBinaryString(tokens: Token[]): string {
  return tokens.map(t => t.toString(2).padStart(5, '0')).join(' ');
}
