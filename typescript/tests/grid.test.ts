/**
 * Binary Grid Database — TypeScript Tests
 * ========================================
 * Core test suite covering encoder, parser, checksum, serialization,
 * arithmetic, grid storage, and geometric queries.
 */

import {
  Token,
  Encoder,
  Parser,
  resolveScaledNumbers,
  computeChecksum, appendChecksum, verifyChecksum,
  packToBytes, unpackFromBytes,
  tokenStreamToBinaryString,
  evaluateParsed,
  DecimalArithmetic,
  BinaryGrid,
  hammingDistance, manhattanDistance, queryByManhattan,
  injectBitFlip, findNextSyncPoint,
  NUMERIC_OPERATORS, NUMERIC_ANNOTATIONS, DIGIT_TO_TOKEN,
} from '../src';

// ── Encoder Tests ─────────────────────────────────────────────────────────

describe('Encoder', () => {
  test('encodes zero', () => {
    expect(Encoder.encodeInteger(0)).toEqual([Token.D0, Token.END]);
  });

  test('encodes positive number', () => {
    expect(Encoder.encodeInteger(123)).toEqual([
      Token.D1, Token.D2, Token.D3, Token.END,
    ]);
  });

  test('encodes negative number', () => {
    expect(Encoder.encodeInteger(-123)).toEqual([
      Token.N1, Token.N2, Token.N3, Token.END,
    ]);
  });

  test('encodes negative number with zeros uses D0', () => {
    // -105: digits are [-1, 0, -5] but 0 uses D0
    expect(Encoder.encodeInteger(-105)).toEqual([
      Token.N1, Token.D0, Token.N5, Token.END,
    ]);
  });

  test('encodes word', () => {
    expect(Encoder.encodeWord('HI')).toEqual([
      Token.START, Token.D7, Token.D8, Token.END,
    ]);
  });

  test('binary string matches spec for 123', () => {
    expect(tokenStreamToBinaryString(Encoder.encodeInteger(123)))
      .toBe('00001 00010 00011 11110');
  });

  test('binary string matches spec for -123', () => {
    expect(tokenStreamToBinaryString(Encoder.encodeInteger(-123)))
      .toBe('10001 10010 10011 11110');
  });

  test('binary string matches spec for word HI', () => {
    expect(tokenStreamToBinaryString(Encoder.encodeWord('HI')))
      .toBe('11111 00111 01000 11110');
  });

  test('encodes expression with digit lists', () => {
    const tokens = Encoder.encodeExpression([[-1, -2, -3], '*', [-8, -1, -7, -5]]);
    expect(tokens).toEqual([
      Token.N1, Token.N2, Token.N3, Token.END,
      Token.T_MUL,
      Token.N8, Token.N1, Token.N7, Token.N5, Token.END,
    ]);
  });

  test('encodes record', () => {
    const tokens = Encoder.encodeRecord(1, 2);
    expect(tokens[tokens.length - 1]).toBe(Token.RECORD);
  });

  test('record binary matches spec', () => {
    const r1 = Encoder.encodeRecord(1);
    const r2 = Encoder.encodeRecord(2);
    const combined = [...r1, ...r2];
    expect(tokenStreamToBinaryString(combined))
      .toBe('00001 11110 11100 00010 11110 11100');
  });
});

// ── Parser Tests ──────────────────────────────────────────────────────────

describe('Parser', () => {
  test('parses simple number', () => {
    const parser = new Parser();
    parser.feedTokens([Token.D1, Token.D2, Token.D3, Token.END]);
    parser.finalize();
    expect(parser.output.length).toBe(1);
    expect(parser.output[0].type).toBe('number');
    expect((parser.output[0] as any).value).toBe(123);
  });

  test('parses negative number', () => {
    const parser = new Parser();
    parser.feedTokens([Token.N1, Token.N2, Token.N3, Token.END]);
    parser.finalize();
    expect((parser.output[0] as any).value).toBe(-123);
  });

  test('parses number then word', () => {
    const parser = new Parser();
    const tokens = [
      ...Encoder.encodeInteger(42),
      ...Encoder.encodeWord('HI'),
    ];
    parser.feedTokens(tokens);
    parser.finalize();
    expect(parser.output.length).toBe(2);
    expect(parser.output[0].type).toBe('number');
    expect(parser.output[1].type).toBe('word');
    expect((parser.output[1] as any).text).toBe('HI');
  });

  test('parses word then number', () => {
    const parser = new Parser();
    const tokens = [
      ...Encoder.encodeWord('HI'),
      ...Encoder.encodeInteger(42),
    ];
    parser.feedTokens(tokens);
    parser.finalize();
    expect((parser.output[0] as any).text).toBe('HI');
    expect((parser.output[1] as any).value).toBe(42);
  });

  test('rejects nested START', () => {
    const parser = new Parser();
    parser.feed(Token.START); // Enter WORD context
    expect(() => parser.feed(Token.START)).toThrow('Nested START');
  });

  test('RECORD creates record', () => {
    const parser = new Parser();
    parser.feedTokens([Token.D1, Token.END, Token.RECORD]);
    parser.finalize();
    expect(parser.records.length).toBe(1);
  });
});

// ── Checksum Tests ────────────────────────────────────────────────────────

describe('Checksum', () => {
  test('computes checksum', () => {
    const tokens = [Token.D1, Token.D2, Token.D3, Token.END];
    // 1 + 2 + 3 + 30 = 36 % 32 = 4
    expect(computeChecksum(tokens)).toBe(4);
  });

  test('verifies passing checksum', () => {
    const tokens = [Token.D1, Token.D2, Token.D3, Token.END];
    const result = verifyChecksum(tokens, 4);
    expect(result.passed).toBe(true);
  });

  test('verifies failing checksum', () => {
    const tokens = [Token.D1, Token.D2, Token.D3, Token.END];
    const result = verifyChecksum(tokens, 5);
    expect(result.passed).toBe(false);
  });

  test('appendChecksum round-trips', () => {
    const data = [
      ...Encoder.encodeInteger(123),
      ...Encoder.encodeInteger(456),
    ];
    const withCs = appendChecksum(data);
    const csIdx = withCs.indexOf(Token.CHECKSUM);
    const segment = withCs.slice(0, csIdx);
    const expected = withCs[csIdx + 1] as number;
    const result = verifyChecksum(segment, expected);
    expect(result.passed).toBe(true);
  });
});

// ── Serialization Tests ───────────────────────────────────────────────────

describe('Serialization', () => {
  test('round-trips tokens', () => {
    const tokens = [Token.D1, Token.D2, Token.D3, Token.END];
    const [packed, pad] = packToBytes(tokens);
    const unpacked = unpackFromBytes(packed, pad);
    expect(unpacked).toEqual(tokens);
  });

  test('round-trips all 32 tokens', () => {
    const tokens = Object.values(Token).filter(t => typeof t === 'number') as Token[];
    const [packed, pad] = packToBytes(tokens);
    const unpacked = unpackFromBytes(packed, pad);
    expect(unpacked).toEqual(tokens);
  });

  test('pad length is correct', () => {
    // 5 tokens = 25 bits → needs 7 bits pad → 4 bytes
    const tokens = Array(5).fill(Token.D1);
    const [packed, pad] = packToBytes(tokens);
    expect(packed.length).toBe(4);
    expect(pad).toBe(7);
  });

  test('8 tokens = no padding', () => {
    const tokens = Array(8).fill(Token.D1);
    const [packed, pad] = packToBytes(tokens);
    expect(packed.length).toBe(5);
    expect(pad).toBe(0);
  });
});

// ── Arithmetic Tests ──────────────────────────────────────────────────────

describe('Arithmetic', () => {
  test('adds numbers', () => {
    const tokens = Encoder.encodeExpression([[2], '+', [3]]);
    const p = new Parser();
    p.feedTokens(tokens);
    p.finalize();
    expect(evaluateParsed(p.output)).toBe(5);
  });

  test('multiplies negative numbers (spec example)', () => {
    const tokens = Encoder.encodeExpression([[-1, -2, -3], '*', [-8, -1, -7, -5]]);
    const p = new Parser();
    p.feedTokens(tokens);
    p.finalize();
    expect(evaluateParsed(p.output)).toBe(1_005_525);
  });

  test('respects operator precedence', () => {
    // 3 + 4 * 2 = 11
    const tokens = Encoder.encodeExpression([[3], '+', [4], '*', [2]]);
    const p = new Parser();
    p.feedTokens(tokens);
    p.finalize();
    expect(evaluateParsed(p.output)).toBe(11);
  });

  test('handles parentheses', () => {
    // (3 + 4) * 2 = 14
    const tokens = Encoder.encodeExpression(['(', [3], '+', [4], ')', '*', [2]]);
    const p = new Parser();
    p.feedTokens(tokens);
    p.finalize();
    expect(evaluateParsed(p.output)).toBe(14);
  });

  test('S operator is NOT an arithmetic operator', () => {
    expect(NUMERIC_OPERATORS.has(Token.T_SCALE)).toBe(false);
    expect(NUMERIC_ANNOTATIONS.has(Token.T_SCALE)).toBe(true);
  });

  test('resolves scaled numbers', () => {
    const tokens = Encoder.encodeExpression([[-1, -2, -3, -4], 'S', [3]]);
    const p = new Parser();
    p.feedTokens(tokens);
    p.finalize();
    const resolved = resolveScaledNumbers(p.output);
    expect(resolved.length).toBe(1);
    expect(resolved[0].type).toBe('scaled');
    const scaled = resolved[0] as any;
    expect(scaled.numerator).toBe(-1234);
    expect(scaled.scale).toBe(3);
    expect(scaled.asFloat).toBeCloseTo(-1.234);
  });
});

// ── Decimal Arithmetic Tests ──────────────────────────────────────────────

describe('DecimalArithmetic', () => {
  test('aligns different scales', () => {
    const a = { type: 'scaled' as const, numerator: 1, scale: 1, asFloat: 0.1 };
    const b = { type: 'scaled' as const, numerator: 2, scale: 2, asFloat: 0.02 };
    const [aa, bb] = DecimalArithmetic.align(a, b);
    expect(aa.numerator).toBe(10);
    expect(aa.scale).toBe(2);
    expect(bb.numerator).toBe(2);
  });

  test('adds 0.1 + 0.02 = 0.12', () => {
    const a = { type: 'scaled' as const, numerator: 1, scale: 1, asFloat: 0.1 };
    const b = { type: 'scaled' as const, numerator: 2, scale: 2, asFloat: 0.02 };
    const result = DecimalArithmetic.add(a, b);
    expect(result.numerator).toBe(12);
    expect(result.scale).toBe(2);
    expect(result.asFloat).toBeCloseTo(0.12);
  });
});

// ── Grid Storage Tests ────────────────────────────────────────────────────

describe('BinaryGrid', () => {
  test('empty grid has zero tokens', () => {
    const grid = new BinaryGrid();
    expect(grid.tokenCount).toBe(0);
    expect(grid.recordCount).toBe(0);
  });

  test('appends record and extracts values', () => {
    const grid = new BinaryGrid();
    const record = grid.appendRecord(Encoder.encodeRecord(42, -5));
    expect(record.valueVector).toEqual([42, -5]);
  });

  test('requires RECORD token', () => {
    const grid = new BinaryGrid();
    expect(() => grid.appendRecord([Token.D1, Token.END]))
      .toThrow('must end with RECORD');
  });

  test('multiple records', () => {
    const grid = new BinaryGrid();
    grid.appendRecord(Encoder.encodeRecord(1));
    grid.appendRecord(Encoder.encodeRecord(2));
    grid.appendRecord(Encoder.encodeRecord(3));
    expect(grid.recordCount).toBe(3);
    expect(grid.getRecord(0).valueVector).toEqual([1]);
    expect(grid.getRecord(2).valueVector).toEqual([3]);
  });

  test('pack/unpack round-trip', () => {
    const grid = new BinaryGrid();
    grid.appendRecord(Encoder.encodeRecord(1, 2));
    grid.appendRecord(Encoder.encodeRecord(-123, 8175));

    const [packed, pad] = grid.pack();
    const restored = BinaryGrid.fromPacked(packed, pad);
    expect(restored.recordCount).toBe(2);
    expect(restored.getRecord(0).valueVector).toEqual([1, 2]);
    expect(restored.getRecord(1).valueVector).toEqual([-123, 8175]);
  });
});

// ── Geometric Query Tests ─────────────────────────────────────────────────

describe('Geometric Queries', () => {
  test('hamming distance same', () => {
    expect(hammingDistance(0b10101, 0b10101)).toBe(0);
  });

  test('hamming distance one bit', () => {
    expect(hammingDistance(0b10101, 0b10100)).toBe(1);
  });

  test('hamming distance all different', () => {
    expect(hammingDistance(0b00000, 0b11111)).toBe(5);
  });

  test('manhattan distance same', () => {
    expect(manhattanDistance([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  test('manhattan distance simple', () => {
    expect(manhattanDistance([0, 0], [3, 4])).toBe(7);
  });

  test('manhattan distance pads shorter with zeros', () => {
    expect(manhattanDistance([1, 2, 3], [1, 2])).toBe(3);
  });

  test('queryByManhattan finds nearby records', () => {
    const grid = new BinaryGrid();
    grid.appendRecord(Encoder.encodeRecord(1, 2, 3));
    grid.appendRecord(Encoder.encodeRecord(2, 3, 4));
    grid.appendRecord(Encoder.encodeRecord(10, 20, 30));
    grid.appendRecord(Encoder.encodeRecord(0, 1, 2));

    const results = queryByManhattan(grid, [1, 2, 3], 10);
    const vectors = results.map(r => r.valueVector);
    expect(vectors).toContainEqual([1, 2, 3]);
    expect(vectors).toContainEqual([2, 3, 4]);
    expect(vectors).toContainEqual([0, 1, 2]);
    expect(vectors).not.toContainEqual([10, 20, 30]);
  });
});

// ── Corruption Detection Tests ────────────────────────────────────────────

describe('Corruption', () => {
  test('injectBitFlip changes token', () => {
    const original = [Token.D1, Token.D2, Token.D3];
    const corrupted = injectBitFlip(original, 1, 0); // Flip MSB of D2
    expect(corrupted[1]).not.toBe(original[1]);
  });

  test('findNextSyncPoint finds RECORD', () => {
    const tokens = [Token.D1, Token.D2, Token.RECORD, Token.D3, Token.END];
    expect(findNextSyncPoint(tokens)).toBe(2);
  });

  test('findNextSyncPoint returns -1 when none', () => {
    const tokens = [Token.D1, Token.D2, Token.D3, Token.END];
    expect(findNextSyncPoint(tokens)).toBe(-1);
  });
});
