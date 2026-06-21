/**
 * Binary Grid Database — Geometric Queries
 * =========================================
 * Hamming distance (address proximity) for shard routing.
 * Manhattan distance (value proximity) for record similarity.
 *
 * These replace traditional SQL JOINs and indexes:
 *   "Which records are nearest to this target in integer space?"
 * instead of
 *   "SELECT * FROM table WHERE foreign_key = X"
 */

import { GridRecord } from './types';
import { BinaryGrid } from './grid';

// ── Hamming Distance ──────────────────────────────────────────────────────

/**
 * Number of bit positions where two addresses differ.
 * Used for shard routing: find the shard whose starting address
 * has the lowest Hamming distance to the target address.
 */
export function hammingDistance(addr1: number, addr2: number): number {
  let xor = addr1 ^ addr2;
  let count = 0;
  while (xor) {
    count += xor & 1;
    xor >>>= 1;
  }
  return count;
}

// ── Manhattan Distance ────────────────────────────────────────────────────

/**
 * Sum of absolute differences between corresponding elements of two vectors.
 * For records of unequal length, pads the shorter with zeros.
 *
 * This is the core query primitive — it replaces SQL WHERE clauses
 * with pure integer arithmetic over record vectors.
 */
export function manhattanDistance(vec1: number[], vec2: number[]): number {
  const n = Math.max(vec1.length, vec2.length);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const a = i < vec1.length ? vec1[i] : 0;
    const b = i < vec2.length ? vec2[i] : 0;
    sum += Math.abs(a - b);
  }
  return sum;
}

// ── Geometric Queries ─────────────────────────────────────────────────────

/**
 * Find all records whose value-vector Manhattan distance from target
 * is less than maxDistance.
 *
 * Equivalent to:
 *   SELECT * WHERE manhattan(value_vector, target) < maxDistance
 * but requires no SQL parser, query planner, or index.
 */
export function queryByManhattan(
  grid: BinaryGrid,
  target: number[],
  maxDistance: number,
): GridRecord[] {
  const results: GridRecord[] = [];
  for (let i = 0; i < grid.recordCount; i++) {
    const record = grid.getRecord(i);
    const dist = manhattanDistance(record.valueVector, target);
    if (dist < maxDistance) {
      results.push(record);
    }
  }
  return results;
}

/**
 * Find the best shard for a target address by minimum Hamming distance.
 * Returns the index of the best shard.
 *
 * No consistent hashing, no ring, no directory service — just POPCNT.
 */
export function queryByHammingShard(
  targetAddress: number,
  shardAddresses: number[],
): number {
  let bestIdx = 0;
  let bestDist = hammingDistance(targetAddress, shardAddresses[0]);
  for (let i = 1; i < shardAddresses.length; i++) {
    const dist = hammingDistance(targetAddress, shardAddresses[i]);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }
  return bestIdx;
}

// ── Corruption Tools ──────────────────────────────────────────────────────

/**
 * Simulate a single bit-flip in a token at the given bit index (0-4).
 */
export function injectBitFlip(tokens: Token[], position: number, bitIndex: number): Token[] {
  const result = [...tokens];
  const oldVal = result[position] as number;
  const newVal = oldVal ^ (1 << (4 - bitIndex));
  result[position] = newVal as Token;
  return result;
}

/**
 * Find the next RECORD or CHECKSUM token (for resynchronization after corruption).
 * Returns the token index, or -1 if none found.
 */
export function findNextSyncPoint(tokens: Token[], start: number = 0): number {
  for (let i = start; i < tokens.length; i++) {
    if (tokens[i] === Token.RECORD || tokens[i] === Token.CHECKSUM) {
      return i;
    }
  }
  return -1;
}
