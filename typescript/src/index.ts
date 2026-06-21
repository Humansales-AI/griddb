/**
 * Binary Grid Database — Public API
 * ==================================
 * A unified 5-bit integer fabric for numbers, words, arithmetic,
 * and geometric queries — now with deterministic TypeScript support.
 *
 * @packageDocumentation
 */

// Types
export {
  Token,
  ParserState,
  ParsedNumber,
  ParsedWord,
  ParsedOperator,
  ParsedScaledNumber,
  ChecksumResult,
  ParsedToken,
  Record,
  GridRecord,
  TOKEN_NAME,
  tokenStreamToBinaryString,
} from './types';

// Token mappings
export {
  NUMERIC_DIGIT_VALUE,
  DIGIT_TO_TOKEN,
  DIGIT_TOKENS,
  NUMERIC_OPERATORS,
  NUMERIC_ANNOTATIONS,
  OPERATOR_SYMBOL,
  SYMBOL_TO_OPERATOR,
  WORD_CHAR,
  CHAR_TO_WORD_TOKEN,
  CONTROL_TOKENS,
  isDigitToken,
  isControlToken,
  isOperator,
  isAnnotation,
} from './tokens';

// Encoder
export { Encoder } from './encoder';

// Parser
export { Parser } from './parser';

// Checksum
export {
  computeChecksum,
  verifyChecksum,
  appendChecksum,
  emitPeriodicChecksums,
} from './checksum';

// Serialization
export {
  packToBytes,
  unpackFromBytes,
  packToHex,
  unpackFromHex,
} from './serialization';

// Arithmetic
export {
  resolveScaledNumbers,
  evaluateParsed,
  DecimalArithmetic,
} from './arithmetic';

// Storage Grid
export { BinaryGrid } from './grid';

// Geometric queries
export {
  hammingDistance,
  manhattanDistance,
  queryByManhattan,
  queryByHammingShard,
  injectBitFlip,
  findNextSyncPoint,
} from './geometry';
