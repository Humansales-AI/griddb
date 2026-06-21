/**
 * Binary Grid Database — Storage Grid
 * ====================================
 * A flat, contiguous, append-only sequence of 5-bit tokens.
 *
 * Storage: bit-addressable, O(1) seek and read.
 * No mandatory file header — the application determines boundaries.
 * Records are delimited by RECORD tokens for geometric query support.
 */

import { Token, ParsedToken, ParsedNumber, GridRecord } from './types';
import { Parser } from './parser';
import { packToBytes, unpackFromBytes } from './serialization';

export class BinaryGrid {
  private tokens: Token[] = [];
  private records: GridRecord[] = [];
  private bitLength: number = 0;

  get tokenCount(): number { return this.tokens.length; }
  get bits(): number { return this.bitLength; }
  get recordCount(): number { return this.records.length; }

  /** Append raw tokens to the grid. Returns the starting bit offset. */
  appendTokens(tokens: Token[]): number {
    const offset = this.bitLength;
    this.tokens.push(...tokens);
    this.bitLength = this.tokens.length * 5;
    return offset;
  }

  /**
   * Append a record (must end with RECORD token).
   * Returns the GridRecord with parsed values for geometric queries.
   */
  appendRecord(tokens: Token[]): GridRecord {
    if (tokens.length === 0 || tokens[tokens.length - 1] !== Token.RECORD) {
      throw new Error('Record must end with RECORD token');
    }

    const offset = this.bitLength;
    this.tokens.push(...tokens);
    this.bitLength = this.tokens.length * 5;

    // Parse to extract values
    const parser = new Parser();
    parser.feedTokens(tokens);
    parser.finalize();
    const numbers = parser.output.filter(
      (p): p is ParsedNumber => p.type === 'number',
    );

    const record: GridRecord = {
      tokens: [...tokens],
      bitOffset: offset,
      bitLength: tokens.length * 5,
      parsedValues: numbers,
      valueVector: numbers.map(n => n.value),
      digitVector: numbers.flatMap(n => n.digits),
    };
    this.records.push(record);
    return record;
  }

  /** Get a record by index. O(1). */
  getRecord(index: number): GridRecord {
    return this.records[index];
  }

  /**
   * Read numTokens starting at the given bit offset. O(1).
   * Offset must be aligned to a 5-bit boundary.
   */
  readAt(bitOffset: number, numTokens: number): Token[] {
    if (bitOffset % 5 !== 0) throw new Error('Bit offset must be aligned to 5-bit boundary');
    const tokenOffset = bitOffset / 5;
    return this.tokens.slice(tokenOffset, tokenOffset + numTokens);
  }

  /** Serialize the entire grid to bytes. Returns [bytes, padLength]. */
  pack(): [Uint8Array, number] {
    return packToBytes(this.tokens);
  }

  /** Deserialize from packed bytes, rebuilding records. */
  static fromPacked(data: Uint8Array, padLength: number): BinaryGrid {
    const tokens = unpackFromBytes(data, padLength);
    const grid = new BinaryGrid();
    grid.tokens = tokens;
    grid.bitLength = tokens.length * 5;

    // Rebuild records by scanning for RECORD tokens
    let recordParser = new Parser();
    let recordStart = 0;

    for (let i = 0; i < tokens.length; i++) {
      recordParser.feed(tokens[i]);
      if (tokens[i] === Token.RECORD) {
        const recordTokens = tokens.slice(recordStart, i + 1);
        const numbers = recordParser.output.filter(
          (p): p is ParsedNumber => p.type === 'number',
        );
        grid.records.push({
          tokens: recordTokens,
          bitOffset: recordStart * 5,
          bitLength: recordTokens.length * 5,
          parsedValues: [...numbers],
          valueVector: numbers.map(n => n.value),
          digitVector: numbers.flatMap(n => n.digits),
        });
        recordParser = new Parser();
        recordStart = i + 1;
      }
    }

    return grid;
  }
}

export { Token, Parser };
