#!/usr/bin/env npx tsx
/**
 * GridDB Cross-Language Conformance Battery — TypeScript
 * ======================================================
 * Encodes 40+ test vectors and outputs JSON with packed hex.
 * Compared against battery_py.py for byte-identical determinism.
 *
 * Run: npx tsx tests/battery_ts.ts > ts_output.json
 */
import { Encoder, packToBytes } from '../src/index';

const TESTS: Record<string, (string | number | bigint)[]> = {
  integers: [0, 1, -1, 7, -5, 42, 123, -123, 999, -9999,
             2147483647, -2147483648, 1000000, 999999999],
  bigIntegers: [12345678901234567890n, 10n**30n],
  bigints: [10n**30n, 999999999999999999999999999999n],
  words: ["A", "Z", "HELLO", "REC", "hello", "aB", "Ba", "abcXYZ",
          "HelloWorld", "test@example.com", "a.b-c", "x@y", "a-b"],
  special: ["a@b", "a-b", "x.y", "a.b-c", "x@y.z"],
  special2: ["a!b", "c#d", "(x)", "a+b", "key=val", "a/b", "[z]", "p;q",
             "a$b", "x%y", "a&b", "x*y", "a_b", "x|y", "a^b", "{x}", "`y`"],
};

const results: Record<string, any[]> = {};
for (const [category, values] of Object.entries(TESTS)) {
  results[category] = [];
  for (const v of values) {
    try {
      let tokens;
      if (typeof v === 'number' || typeof v === 'bigint') {
        tokens = Encoder.encodeInteger(v);
      } else {
        tokens = Encoder.encodeWord(v);
      }
      const [packed, pad] = packToBytes(tokens);
      results[category].push({
        input: String(v),
        hex: Buffer.from(packed).toString('hex'),
        tokens: tokens.length,
        pad,
      });
    } catch (e: any) {
      results[category].push({ input: String(v), error: e.message });
    }
  }
}

console.log(JSON.stringify(results, null, 2));
