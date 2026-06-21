/**
 * Binary Grid Database — Serialization
 * =====================================
 * 5-bit ↔ 8-bit packing for storage and transmission.
 *
 * Tokens are 5 bits each.  Physical storage is byte-addressed.
 * The packer concatenates 5-bit codes into a bitstream, pads to
 * a byte boundary, and converts to a Uint8Array.  The pad length
 * is tracked for lossless unpacking.
 */

import { Token } from './types';

/**
 * Pack a list of 5-bit tokens into a byte array.
 * Returns [bytes, padLength] where padLength is the number of zero bits
 * padded to reach a byte boundary (0-7).
 */
export function packToBytes(tokens: Token[]): [Uint8Array, number] {
  // Build bit array
  const bits: number[] = [];
  for (const t of tokens) {
    const val = t as number;
    for (let i = 4; i >= 0; i--) {
      bits.push((val >> i) & 1);
    }
  }

  // Pad to byte boundary
  const padLength = (8 - (bits.length % 8)) % 8;
  for (let i = 0; i < padLength; i++) bits.push(0);

  // Pack into bytes
  const byteCount = bits.length / 8;
  const bytes = new Uint8Array(byteCount);
  for (let i = 0; i < byteCount; i++) {
    let byteVal = 0;
    for (let j = 0; j < 8; j++) {
      byteVal = (byteVal << 1) | bits[i * 8 + j];
    }
    bytes[i] = byteVal;
  }

  return [bytes, padLength];
}

/**
 * Unpack bytes back into 5-bit tokens.
 *
 * @param data - The byte array to unpack.
 * @param padLength - Number of zero bits padded at the end (from packToBytes).
 * @param numTokens - If specified, unpack exactly this many tokens.
 */
export function unpackFromBytes(
  data: Uint8Array,
  padLength: number = 0,
  numTokens?: number,
): Token[] {
  // Convert bytes to bit stream
  const bits: number[] = [];
  for (const byte of data) {
    for (let i = 7; i >= 0; i--) {
      bits.push((byte >> i) & 1);
    }
  }

  // Remove padding
  const trimmed = bits.slice(0, bits.length - padLength);

  // Extract 5-bit tokens
  const tokens: Token[] = [];
  for (let i = 0; i + 5 <= trimmed.length; i += 5) {
    if (numTokens !== undefined && tokens.length >= numTokens) break;
    let val = 0;
    for (let j = 0; j < 5; j++) {
      val = (val << 1) | trimmed[i + j];
    }
    tokens.push(val as Token);
  }

  return tokens;
}

/**
 * Pack tokens to a hex string (for display/debugging).
 */
export function packToHex(tokens: Token[]): string {
  const [bytes, _pad] = packToBytes(tokens);
  return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Unpack from a hex string.
 */
export function unpackFromHex(hex: string, padLength: number = 0): Token[] {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return unpackFromBytes(bytes, padLength);
}
