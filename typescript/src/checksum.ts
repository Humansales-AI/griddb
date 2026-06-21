/**
 * Binary Grid Database — Checksum
 * ================================
 * Modulo-32 checksum for corruption detection and resynchronization.
 *
 * The CHECKSUM token (11101) marks the beginning of a checksum block.
 * The following 5 bits contain a modulo-32 checksum of all previous
 * tokens since the last CHECKSUM (or start of stream).
 */

import { Token, ChecksumResult } from './types';

/**
 * Compute modulo-32 checksum: sum of all token integer values % 32.
 */
export function computeChecksum(tokens: Token[]): number {
  const total = tokens.reduce((sum, t) => sum + (t as number), 0);
  return total % 32;
}

/**
 * Verify that a checksum value matches the computed checksum of tokens.
 */
export function verifyChecksum(tokens: Token[], expected: number): ChecksumResult {
  const computed = computeChecksum(tokens);
  return {
    type: 'checksum',
    expected,
    computed,
    passed: computed === expected,
  };
}

/**
 * Append a CHECKSUM token + checksum value to a token list.
 */
export function appendChecksum(tokens: Token[]): Token[] {
  const csValue = computeChecksum(tokens);
  const result = [...tokens];
  result.push(Token.CHECKSUM);
  // Find a token whose integer value equals csValue
  const csToken = tokenForValue(csValue);
  result.push(csToken);
  return result;
}

/** Find a token whose integer value equals the given value. */
function tokenForValue(value: number): Token {
  for (const t of Object.values(Token)) {
    if (typeof t === 'number' && t === value) return t as Token;
  }
  throw new Error(`No token with value ${value}`);
}

/**
 * Emit a CHECKSUM marker every `interval` records.
 */
export function emitPeriodicChecksums(
  recordTokensList: Token[][],
  interval: number = 10,
): Token[] {
  const result: Token[] = [];
  let segment: Token[] = [];

  for (let i = 0; i < recordTokensList.length; i++) {
    segment.push(...recordTokensList[i]);
    if ((i + 1) % interval === 0 || i === recordTokensList.length - 1) {
      result.push(...appendChecksum(segment));
      segment = [];
    } else {
      result.push(...recordTokensList[i]);
    }
  }

  return result;
}
