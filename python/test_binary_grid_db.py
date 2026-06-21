#!/usr/bin/env python3
"""
Comprehensive test suite for the Binary Grid Database.

Covers:
  - Token definitions & mapping consistency
  - Encoder (numbers, words, expressions, records)
  - Parser state machine (all transitions)
  - Checksum computation & verification
  - Serialization (5-bit ↔ 8-bit round-trips)
  - Arithmetic evaluator (all operators, precedence, parentheses)
  - Scale operator
  - BinaryGrid storage (append, read, records)
  - Geometric queries (Hamming, Manhattan)
  - Corruption detection & recovery
  - All spec examples (Appendix B)
  - Edge cases & adversarial inputs
"""

import unittest
import sys
import os

# Import the module under test
from binary_grid_db import (
    Token,
    Encoder,
    Parser,
    ParserState,
    ParsedNumber,
    ParsedScaledNumber,
    ParsedWord,
    ParsedOperator,
    ChecksumResult,
    Record,
    compute_checksum,
    verify_checksum,
    append_checksum,
    pack_to_bytes,
    unpack_from_bytes,
    token_stream_to_binary_string,
    ArithmeticEvaluator,
    resolve_scaled_numbers,
    DecimalArithmetic,
    BinaryGrid,
    BinaryGridDB,
    GridRecord,
    hamming_distance,
    manhattan_distance,
    query_by_manhattan,
    query_by_hamming_shard,
    inject_bit_flip,
    find_next_sync_point,
    scan_for_corruption,
    NUMERIC_DIGIT_VALUE,
    DIGIT_TO_TOKEN,
    NUMERIC_OPERATORS,
    NUMERIC_ANNOTATIONS,
    OPERATOR_SYMBOL,
    SYMBOL_TO_OPERATOR,
    WORD_CHAR,
    CHAR_TO_WORD_TOKEN,
    CONTROL_TOKENS,
    DIGIT_TOKENS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TOKEN DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenDefinitions(unittest.TestCase):
    """Verify the 32-slot vocabulary is consistent and complete."""

    def test_all_32_slots_defined(self):
        """Every 5-bit value (0-31) must map to exactly one Token."""
        values = set(int(t) for t in Token)
        self.assertEqual(len(values), 32, "Must have exactly 32 unique token values")
        self.assertEqual(min(values), 0)
        self.assertEqual(max(values), 31)

    def test_control_tokens_correct_values(self):
        """Control tokens must occupy 0x1C-0x1F (28-31)."""
        self.assertEqual(int(Token.RECORD), 0b11100)
        self.assertEqual(int(Token.CHECKSUM), 0b11101)
        self.assertEqual(int(Token.END), 0b11110)
        self.assertEqual(int(Token.START), 0b11111)

    def test_positive_digit_mapping(self):
        """Positive digits 0-9 map to 0x00-0x09."""
        for d in range(10):
            token = DIGIT_TO_TOKEN[d]
            self.assertEqual(int(token), d, f"Digit {d} should map to {d:05b}")

    def test_negative_digit_mapping(self):
        """Negative digits -1 to -9 map to 0x11-0x19 (leading bit toggled from positive)."""
        for d in range(1, 10):
            token = DIGIT_TO_TOKEN[-d]
            expected = 0b10000 | d
            self.assertEqual(int(token), expected,
                             f"Digit -{d} should map to {expected:05b}")

    def test_digit_roundtrip(self):
        """Every digit token round-trips through NUMERIC_DIGIT_VALUE."""
        for token, value in NUMERIC_DIGIT_VALUE.items():
            if value is not None:
                self.assertEqual(DIGIT_TO_TOKEN[value], token)

    def test_operator_symbol_roundtrip(self):
        """Every operator symbol round-trips through SYMBOL_TO_OPERATOR."""
        for token, symbol in OPERATOR_SYMBOL.items():
            self.assertEqual(SYMBOL_TO_OPERATOR[symbol], token)

    def test_word_char_roundtrip(self):
        """Every word character round-trips through CHAR_TO_WORD_TOKEN."""
        for token, char in WORD_CHAR.items():
            self.assertEqual(CHAR_TO_WORD_TOKEN[char], token)

    def test_letter_coverage(self):
        """All 26 uppercase letters are represented in WORD context."""
        letters = set(WORD_CHAR.values())
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.assertIn(ch, letters, f"Letter '{ch}' missing from WORD_CHAR")

    def test_no_overlap_digit_vs_control(self):
        """Digit tokens must not overlap with control tokens."""
        for token in DIGIT_TOKENS:
            self.assertNotIn(token, CONTROL_TOKENS,
                             f"{token.name} is both digit and control")

    def test_no_overlap_operator_vs_control(self):
        """Operator tokens must not overlap with control tokens."""
        for token in NUMERIC_OPERATORS:
            self.assertNotIn(token, CONTROL_TOKENS,
                             f"{token.name} is both operator and control")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncoder(unittest.TestCase):
    """Verify the Encoder produces correct 5-bit token streams."""

    # ── Integer encoding ──

    def test_encode_zero(self):
        tokens = Encoder.encode_integer(0)
        self.assertEqual(tokens, [Token.D0, Token.END])

    def test_encode_positive_single_digit(self):
        tokens = Encoder.encode_integer(7)
        self.assertEqual(tokens, [Token.D7, Token.END])

    def test_encode_negative_single_digit(self):
        tokens = Encoder.encode_integer(-5)
        self.assertEqual(tokens, [Token.N5, Token.END])

    def test_encode_positive_multi_digit(self):
        tokens = Encoder.encode_integer(123)
        self.assertEqual(tokens, [Token.D1, Token.D2, Token.D3, Token.END])

    def test_encode_negative_multi_digit(self):
        tokens = Encoder.encode_integer(-123)
        self.assertEqual(tokens, [Token.N1, Token.N2, Token.N3, Token.END])

    def test_encode_large_number(self):
        tokens = Encoder.encode_integer(987654321)
        expected = [Token.D9, Token.D8, Token.D7, Token.D6, Token.D5,
                    Token.D4, Token.D3, Token.D2, Token.D1, Token.END]
        self.assertEqual(tokens, expected)

    def test_encode_negative_large_number(self):
        tokens = Encoder.encode_integer(-8175)
        expected = [Token.N8, Token.N1, Token.N7, Token.N5, Token.END]
        self.assertEqual(tokens, expected)

    def test_binary_string_matches_spec_examples(self):
        """Verify the exact binary strings from the specification Appendix B."""
        # Number 123: [1,2,3] END
        self.assertEqual(
            token_stream_to_binary_string(Encoder.encode_integer(123)),
            "00001 00010 00011 11110"
        )
        # Number -123: [-1,-2,-3] END
        self.assertEqual(
            token_stream_to_binary_string(Encoder.encode_integer(-123)),
            "10001 10010 10011 11110"
        )

    # ── Word encoding ──

    def test_encode_simple_word(self):
        tokens = Encoder.encode_word("HI")
        self.assertEqual(tokens, [Token.START, Token.D7, Token.D8, Token.END])

    def test_encode_word_with_space(self):
        tokens = Encoder.encode_word("A B")
        self.assertEqual(tokens, [
            Token.START, Token.D0, Token.T_POW,  # T_POW = SPACE in WORD
            Token.D1, Token.END
        ])

    def test_encode_word_binary_matches_spec(self):
        """Word 'HI': START [H][I] END → 11111 00111 01000 11110"""
        self.assertEqual(
            token_stream_to_binary_string(Encoder.encode_word("HI")),
            "11111 00111 01000 11110"
        )

    def test_encode_word_hello(self):
        tokens = Encoder.encode_word("HELLO")
        expected = [Token.START,
                    Token.D7, Token.D4, Token.T_MINUS, Token.T_MINUS, Token.T_EQ,
                    Token.END]
        self.assertEqual(tokens, expected)

    def test_encode_word_lowercase(self):
        """Lowercase input uses SPECIAL context (START-in-WORD)."""
        tokens = Encoder.encode_word("hi")
        # START → START(enter SPECIAL) → h → i → END(pop to WORD) → END(pop to NUM)
        self.assertEqual(tokens, [
            Token.START, Token.START,   # WORD then SPECIAL
            Token.D7, Token.D8,          # h, i in SPECIAL
            Token.END, Token.END,        # Pop SPECIAL, Pop WORD
        ])

    def test_encode_word_invalid_char_raises(self):
        """Characters outside A-Z, space, period should raise ValueError."""
        with self.assertRaises(ValueError):
            Encoder.encode_word("HI!")

    def test_encode_word_with_period(self):
        tokens = Encoder.encode_word("OK.")
        self.assertEqual(tokens[-2], Token.T_SCALE)  # T_SCALE = '.' in WORD

    # ── Expression encoding ──

    def test_encode_expression_single_number(self):
        tokens = Encoder.encode_expression([42])
        self.assertEqual(tokens, [Token.D4, Token.D2, Token.END])

    def test_encode_expression_multi_digit_via_list(self):
        """[-1, -2, -3] should encode as multi-digit number, not three singles."""
        tokens = Encoder.encode_expression([[-1, -2, -3]])
        self.assertEqual(tokens, [Token.N1, Token.N2, Token.N3, Token.END])

    def test_encode_expression_with_operator(self):
        """[-1,-2,-3] * [-8,-1,-7,-5]"""
        tokens = Encoder.encode_expression([[-1, -2, -3], '*', [-8, -1, -7, -5]])
        expected = [
            Token.N1, Token.N2, Token.N3, Token.END,
            Token.T_MUL,
            Token.N8, Token.N1, Token.N7, Token.N5, Token.END,
        ]
        self.assertEqual(tokens, expected)

    def test_encode_expression_binary_matches_spec(self):
        """Expression 2 + (-3): [2][+][-3] → 00010 11110 01010 10011 11110"""
        tokens = Encoder.encode_expression([[2], '+', [-3]])
        self.assertEqual(
            token_stream_to_binary_string(tokens),
            "00010 11110 01010 10011 11110"
        )

    def test_encode_expression_scale_matches_spec(self):
        """Scale -1.23: [-1,-2,-3] S [2] → 10001 10010 10011 11110 11011 00010 11110"""
        tokens = Encoder.encode_expression([[-1, -2, -3], 'S', [2]])
        self.assertEqual(
            token_stream_to_binary_string(tokens),
            "10001 10010 10011 11110 11011 00010 11110"
        )

    # ── Record encoding ──

    def test_encode_record_single_value(self):
        tokens = Encoder.encode_record(42)
        self.assertEqual(tokens[-1], Token.RECORD)
        self.assertEqual(len(tokens), 4)  # D4, D2, END, RECORD

    def test_encode_record_multiple_values(self):
        tokens = Encoder.encode_record(1, "HI", -5)
        self.assertEqual(tokens[-1], Token.RECORD)
        self.assertIn(Token.START, tokens)  # Word context activated
        self.assertIn(Token.END, tokens)

    def test_encode_record_binary_matches_spec(self):
        """Record (1, 2): [1] RECORD [2] RECORD → 00001 11110 11100 00010 11110 11100"""
        r1 = Encoder.encode_record(1)
        r2 = Encoder.encode_record(2)
        combined = r1 + r2
        self.assertEqual(
            token_stream_to_binary_string(combined),
            "00001 11110 11100 00010 11110 11100"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARSER STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestParser(unittest.TestCase):
    """Verify the finite-state machine correctly transitions and accumulates."""

    def setUp(self):
        self.parser = Parser()

    def test_initial_state_is_num(self):
        self.assertEqual(self.parser.state, ParserState.NUM)

    def test_parse_simple_number(self):
        tokens = [Token.D1, Token.D2, Token.D3, Token.END]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        self.assertEqual(len(self.parser.output), 1)
        num = self.parser.output[0]
        self.assertIsInstance(num, ParsedNumber)
        self.assertEqual(num.value, 123)
        self.assertEqual(num.digits, [1, 2, 3])

    def test_parse_negative_number(self):
        tokens = [Token.N1, Token.N2, Token.N3, Token.END]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        num = self.parser.output[0]
        self.assertIsInstance(num, ParsedNumber)
        self.assertEqual(num.value, -123)
        self.assertEqual(num.digits, [-1, -2, -3])

    def test_parse_zero(self):
        tokens = [Token.D0, Token.END]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        self.assertEqual(self.parser.output[0].value, 0)

    def test_parse_mixed_signs_in_number(self):
        """Can a number have mixed sign digits? Spec says each digit is signed.
        e.g., [-1, 2, 3] should be -1*100 + 2*10 + 3 = -100+20+3 = -77"""
        tokens = [Token.N1, Token.D2, Token.D3, Token.END]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        num = self.parser.output[0]
        # -1*100 + 2*10 + 3 = -100 + 20 + 3 = -77
        self.assertEqual(num.value, -77)

    def test_parse_simple_word(self):
        tokens = [Token.START, Token.D7, Token.D8, Token.END]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        self.assertEqual(len(self.parser.output), 1)
        word = self.parser.output[0]
        self.assertIsInstance(word, ParsedWord)
        self.assertEqual(word.text, "HI")

    def test_parse_word_with_space(self):
        tokens = [Token.START, Token.D7, Token.T_POW, Token.D8, Token.END]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        word = self.parser.output[0]
        self.assertEqual(word.text, "H I")

    def test_parse_number_then_word(self):
        """NUM(42) then WORD('HI') — state transitions correctly."""
        tokens = (Encoder.encode_integer(42) +
                  Encoder.encode_word("HI"))
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        self.assertEqual(len(self.parser.output), 2)
        self.assertIsInstance(self.parser.output[0], ParsedNumber)
        self.assertIsInstance(self.parser.output[1], ParsedWord)
        self.assertEqual(self.parser.output[0].value, 42)
        self.assertEqual(self.parser.output[1].text, "HI")

    def test_parse_word_then_number(self):
        """WORD('HI') then NUM(42) — state returns to NUM."""
        tokens = (Encoder.encode_word("HI") +
                  Encoder.encode_integer(42))
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        self.assertEqual(len(self.parser.output), 2)
        self.assertIsInstance(self.parser.output[0], ParsedWord)
        self.assertIsInstance(self.parser.output[1], ParsedNumber)
        self.assertEqual(self.parser.output[1].value, 42)

    def test_parse_operator_between_numbers(self):
        """NUM(2) OP(+) NUM(3)"""
        tokens = [
            Token.D2, Token.END,
            Token.T_PLUS,
            Token.D3, Token.END,
        ]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        self.assertEqual(len(self.parser.output), 3)  # NUM, OP, NUM
        self.assertIsInstance(self.parser.output[0], ParsedNumber)
        self.assertIsInstance(self.parser.output[1], ParsedOperator)
        self.assertIsInstance(self.parser.output[2], ParsedNumber)
        self.assertEqual(self.parser.output[1].symbol, '+')

    def test_record_boundary_emits_token(self):
        """RECORD should be emitted as a control token."""
        tokens = [Token.D1, Token.END, Token.RECORD]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        # Output should contain the number and the RECORD marker
        self.assertTrue(any(isinstance(p, ParsedNumber) for p in self.parser.output))
        self.assertIn(Token.RECORD, self.parser.output)

    def test_record_creates_record_object(self):
        """RECORD should group tokens into a Record."""
        tokens = [Token.D1, Token.END, Token.RECORD]
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        self.assertEqual(len(self.parser.records), 1)
        self.assertIsInstance(self.parser.records[0], Record)

    def test_start_in_word_enters_special(self):
        """START inside WORD context now enters SPECIAL context (lowercase + special chars)."""
        self.parser.state = ParserState.WORD
        result = self.parser.feed(Token.START)
        self.assertEqual(self.parser.state, ParserState.SPECIAL)
        self.assertIsNone(result)  # START consumes without emitting

    def test_end_in_word_switches_to_num(self):
        """END in WORD state should finalize word and switch to NUM."""
        self.parser.state = ParserState.WORD
        self.parser.accumulator = [int(Token.D7), int(Token.D8)]
        self.parser.feed(Token.END)
        self.parser.finalize()
        self.assertEqual(self.parser.state, ParserState.NUM)
        self.assertEqual(self.parser.output[0].text, "HI")

    def test_record_in_word_switches_to_num(self):
        """RECORD in WORD state finalizes word, emits RECORD, switches to NUM."""
        self.parser.state = ParserState.WORD
        self.parser.accumulator = [int(Token.D1)]
        self.parser.feed(Token.RECORD)
        self.assertEqual(self.parser.state, ParserState.NUM)
        # Word 'B' should be emitted
        self.assertTrue(any(
            isinstance(p, ParsedWord) and p.text == 'B'
            for p in self.parser.output
        ))

    def test_checksum_in_word_switches_to_num(self):
        """CHECKSUM in WORD state finalizes word, switches to NUM."""
        self.parser.state = ParserState.WORD
        self.parser.accumulator = [int(Token.D2)]
        result = self.parser.feed(Token.CHECKSUM)
        self.assertEqual(self.parser.state, ParserState.NUM)
        self.assertEqual(result, Token.CHECKSUM)

    def test_parse_expression_full(self):
        """Parse -123 * -8175 expression."""
        tokens = Encoder.encode_expression([[-1, -2, -3], '*', [-8, -1, -7, -5]])
        self.parser.feed_tokens(tokens)
        self.parser.finalize()
        self.assertEqual(len(self.parser.output), 3)
        self.assertEqual(self.parser.output[0].value, -123)
        self.assertEqual(self.parser.output[1].symbol, '*')
        self.assertEqual(self.parser.output[2].value, -8175)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CHECKSUM
# ═══════════════════════════════════════════════════════════════════════════════

class TestChecksum(unittest.TestCase):
    """Verify modulo-32 checksum computation, verification, and corruption detection."""

    def test_compute_simple(self):
        tokens = [Token.D1, Token.D2, Token.D3, Token.END]
        # 1 + 2 + 3 + 30 = 36; 36 % 32 = 4
        self.assertEqual(compute_checksum(tokens), 4)

    def test_compute_empty(self):
        self.assertEqual(compute_checksum([]), 0)

    def test_compute_single_token(self):
        self.assertEqual(compute_checksum([Token.D0]), 0)

    def test_verify_pass(self):
        tokens = [Token.D1, Token.D2, Token.D3, Token.END]
        result = verify_checksum(tokens, 4)
        self.assertTrue(result.passed)
        self.assertEqual(result.expected, 4)
        self.assertEqual(result.computed, 4)

    def test_verify_fail(self):
        tokens = [Token.D1, Token.D2, Token.D3, Token.END]
        result = verify_checksum(tokens, 99)
        self.assertFalse(result.passed)

    def test_append_checksum(self):
        tokens = [Token.D1, Token.END]
        result = append_checksum(tokens)
        self.assertEqual(result[-2], Token.CHECKSUM)
        # The last token should be the checksum value
        cs_value = int(result[-1])
        self.assertEqual(cs_value, compute_checksum(tokens))

    def test_checksum_roundtrip(self):
        """Append checksum, then verify it passes."""
        data = Encoder.encode_integer(123) + Encoder.encode_integer(456)
        with_checksum = append_checksum(data)
        # Extract the checksum segment
        cs_idx = with_checksum.index(Token.CHECKSUM)
        segment = with_checksum[:cs_idx]
        expected = int(with_checksum[cs_idx + 1])
        result = verify_checksum(segment, expected)
        self.assertTrue(result.passed)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSerialization(unittest.TestCase):
    """Verify 5-bit ↔ 8-bit packing/unpacking is lossless."""

    def test_roundtrip_single_token(self):
        tokens = [Token.D5]
        packed, pad = pack_to_bytes(tokens)
        unpacked = unpack_from_bytes(packed, pad)
        self.assertEqual(tokens, unpacked)

    def test_roundtrip_multiple_tokens(self):
        tokens = [Token.D1, Token.D2, Token.D3, Token.END]
        packed, pad = pack_to_bytes(tokens)
        unpacked = unpack_from_bytes(packed, pad)
        self.assertEqual(tokens, unpacked)

    def test_roundtrip_full_expression(self):
        tokens = Encoder.encode_expression([[-1, -2, -3], '*', [-8, -1, -7, -5]])
        packed, pad = pack_to_bytes(tokens)
        unpacked = unpack_from_bytes(packed, pad)
        self.assertEqual(tokens, unpacked)

    def test_roundtrip_with_words(self):
        tokens = (Encoder.encode_integer(42) +
                  Encoder.encode_word("HELLO") +
                  Encoder.encode_integer(-7))
        packed, pad = pack_to_bytes(tokens)
        unpacked = unpack_from_bytes(packed, pad)
        self.assertEqual(tokens, unpacked)

    def test_roundtrip_with_records(self):
        tokens = (Encoder.encode_record(1, 2) +
                  Encoder.encode_record(-123, 8175))
        packed, pad = pack_to_bytes(tokens)
        unpacked = unpack_from_bytes(packed, pad)
        self.assertEqual(tokens, unpacked)

    def test_roundtrip_all_32_tokens(self):
        """Every token value should survive serialization."""
        tokens = list(Token)
        packed, pad = pack_to_bytes(tokens)
        unpacked = unpack_from_bytes(packed, pad)
        self.assertEqual(tokens, unpacked)

    def test_pad_length_boundaries(self):
        """Different token counts produce correct pad lengths."""
        for n in range(1, 33):
            tokens = [Token.D1] * n
            packed, pad = pack_to_bytes(tokens)
            self.assertGreaterEqual(pad, 0)
            self.assertLess(pad, 8)
            # Total bits should be a multiple of 8
            total_bits = n * 5 + pad
            self.assertEqual(total_bits % 8, 0)

    def test_num_tokens_parameter(self):
        """The num_tokens parameter limits unpacking."""
        tokens = [Token.D1, Token.D2, Token.D3, Token.D4, Token.D5]
        packed, pad = pack_to_bytes(tokens)
        unpacked = unpack_from_bytes(packed, pad, num_tokens=3)
        self.assertEqual(len(unpacked), 3)
        self.assertEqual(unpacked, tokens[:3])

    def test_packed_size_efficiency(self):
        """5 tokens (25 bits) → 4 bytes (32 bits) with 7 pad bits."""
        tokens = [Token.D1] * 5
        packed, pad = pack_to_bytes(tokens)
        self.assertEqual(len(packed), 4)
        self.assertEqual(pad, 7)

    def test_packed_size_efficiency_8_tokens(self):
        """8 tokens (40 bits) → exactly 5 bytes, no padding."""
        tokens = [Token.D1] * 8
        packed, pad = pack_to_bytes(tokens)
        self.assertEqual(len(packed), 5)
        self.assertEqual(pad, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ARITHMETIC EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestArithmeticEvaluator(unittest.TestCase):
    """Verify Shunting-Yard + RPN evaluation of all operators."""

    def _eval(self, *items) -> int:
        """Helper: encode expression, parse it, evaluate it."""
        tokens = Encoder.encode_expression(list(items))
        parser = Parser()
        parser.feed_tokens(tokens)
        parser.finalize()
        return ArithmeticEvaluator.evaluate_parsed(parser.output)

    def test_addition(self):
        self.assertEqual(self._eval([2], '+', [3]), 5)

    def test_subtraction(self):
        self.assertEqual(self._eval([7], '-', [4]), 3)

    def test_multiplication(self):
        self.assertEqual(self._eval([6], '*', [7]), 42)

    def test_division(self):
        self.assertEqual(self._eval(10, '/', [3]), 3)  # Integer division

    def test_exponentiation(self):
        self.assertEqual(self._eval([2], '^', 10), 1024)

    def test_equality_true(self):
        self.assertEqual(self._eval([5], '=', [5]), 1)

    def test_equality_false(self):
        self.assertEqual(self._eval([5], '=', [3]), 0)

    def test_negative_operands(self):
        self.assertEqual(self._eval([-5], '+', [3]), -2)

    def test_multi_digit_operands(self):
        # -123 * -8175 = 1,005,525
        self.assertEqual(
            self._eval([-1, -2, -3], '*', [-8, -1, -7, -5]),
            1_005_525
        )

    def test_operator_precedence(self):
        """3 + 4 * 2 = 3 + 8 = 11 (not 7 * 2 = 14)"""
        self.assertEqual(self._eval([3], '+', [4], '*', [2]), 11)

    def test_operator_precedence_division(self):
        """10 - 6 / 2 = 10 - 3 = 7"""
        self.assertEqual(self._eval([1, 0], '-', [6], '/', [2]), 7)

    def test_parentheses(self):
        """(3 + 4) * 2 = 14"""
        result = self._eval('(', [3], '+', [4], ')', '*', [2])
        self.assertEqual(result, 14)

    def test_nested_parentheses(self):
        """((2 + 3) * (4 - 1)) = 5 * 3 = 15"""
        result = self._eval(
            '(', '(', [2], '+', [3], ')', '*', '(', [4], '-', [1], ')', ')'
        )
        self.assertEqual(result, 15)

    def test_right_associative_exponentiation(self):
        """2^3^2 = 2^(3^2) = 2^9 = 512 (not (2^3)^2 = 64)"""
        result = self._eval([2], '^', [3], '^', [2])
        self.assertEqual(result, 512)

    def test_scale_annotation_storage(self):
        """S is a storage annotation, NOT an arithmetic operator.
        -1234 S 3 means 'integer -1234, 3 implied decimal places' → -1.234
        The database stores pure integers. The application layer interprets scale.
        """
        # Encode: [-1,-2,-3,-4, END, S, 3, END]
        tokens = Encoder.encode_expression([[-1, -2, -3, -4], 'S', [3]])
        parser = Parser()
        parser.feed_tokens(tokens)
        parser.finalize()
        # Before resolution: NUM(-1234), OP('S'), NUM(3)
        self.assertEqual(len(parser.output), 3)
        self.assertEqual(parser.output[0].value, -1234)
        self.assertEqual(parser.output[1].symbol, 'S')
        self.assertEqual(parser.output[2].value, 3)
        # After resolution: a single SCALED(-1234 / 10^3)
        resolved = resolve_scaled_numbers(parser.output)
        self.assertEqual(len(resolved), 1)
        self.assertIsInstance(resolved[0], ParsedScaledNumber)
        self.assertEqual(resolved[0].numerator, -1234)
        self.assertEqual(resolved[0].scale, 3)
        self.assertAlmostEqual(resolved[0].as_float, -1.234)

    def test_scale_annotation_positive(self):
        """5 S 2 → integer 5, 2 decimal places → 0.05"""
        tokens = Encoder.encode_expression([[5], 'S', [2]])
        parser = Parser()
        parser.feed_tokens(tokens)
        parser.finalize()
        resolved = resolve_scaled_numbers(parser.output)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].numerator, 5)
        self.assertEqual(resolved[0].scale, 2)
        self.assertAlmostEqual(resolved[0].as_float, 0.05)

    def test_scale_not_in_arithmetic_operators(self):
        """S must not be in the set of arithmetic operators."""
        self.assertNotIn(Token.T_SCALE, NUMERIC_OPERATORS)
        self.assertIn(Token.T_SCALE, NUMERIC_ANNOTATIONS)

    def test_mismatched_parentheses_raises(self):
        with self.assertRaises(ValueError):
            self._eval('(', [1], '+', [2])

    def test_division_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            self._eval([5], '/', [0])

    def test_expression_from_spec(self):
        """The exact expression from the specification: -123 * -8175"""
        tokens = Encoder.encode_expression([[-1, -2, -3], '*', [-8, -1, -7, -5]])
        parser = Parser()
        parser.feed_tokens(tokens)
        parser.finalize()
        result = ArithmeticEvaluator.evaluate_parsed(parser.output)
        self.assertEqual(result, 1_005_525)

    def test_chained_addition(self):
        self.assertEqual(self._eval([1], '+', [2], '+', [3], '+', [4]), 10)

    def test_chained_multiplication(self):
        self.assertEqual(self._eval([2], '*', [3], '*', [4]), 24)

    def test_complex_expression(self):
        """3 + 4 * 2 / (1 - 5) ^ 2 = 3 + 8 / 16 = 3 + 0 = 3"""
        result = self._eval(
            [3], '+', [4], '*', [2], '/', '(', [1], '-', [5], ')', '^', [2]
        )
        # 3 + 4*2/(1-5)^2 = 3 + 8/(-4)^2 = 3 + 8/16 = 3 + 0 = 3
        self.assertEqual(result, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BINARY GRID STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinaryGrid(unittest.TestCase):
    """Verify the append-only bit-addressable grid."""

    def setUp(self):
        self.grid = BinaryGrid()

    def test_empty_grid(self):
        self.assertEqual(self.grid.token_count, 0)
        self.assertEqual(self.grid.bit_length, 0)
        self.assertEqual(self.grid.record_count, 0)

    def test_append_tokens(self):
        offset = self.grid.append_tokens([Token.D1, Token.END])
        self.assertEqual(offset, 0)
        self.assertEqual(self.grid.token_count, 2)
        self.assertEqual(self.grid.bit_length, 10)

    def test_append_tokens_returns_correct_offset(self):
        self.grid.append_tokens([Token.D1])
        offset = self.grid.append_tokens([Token.D2])
        self.assertEqual(offset, 5)  # after first token

    def test_append_record(self):
        tokens = Encoder.encode_record(1, 2)
        record = self.grid.append_record(tokens)
        self.assertEqual(record.bit_offset, 0)
        self.assertEqual(record.bit_length, len(tokens) * 5)
        self.assertEqual(self.grid.record_count, 1)

    def test_append_record_requires_record_token(self):
        with self.assertRaises(ValueError):
            self.grid.append_record([Token.D1, Token.END])  # No RECORD

    def test_append_record_extracts_values(self):
        tokens = Encoder.encode_record(42, -5)
        record = self.grid.append_record(tokens)
        self.assertEqual(len(record.parsed_values), 2)
        self.assertEqual(record.parsed_values[0].value, 42)
        self.assertEqual(record.parsed_values[1].value, -5)

    def test_append_record_value_vector(self):
        tokens = Encoder.encode_record(1, 2, 3)
        record = self.grid.append_record(tokens)
        self.assertEqual(record.value_vector, [1, 2, 3])

    def test_append_record_digit_vector(self):
        """For a record with multi-digit numbers, digit_vector flattens all digits."""
        tokens = Encoder.encode_record(-123)
        record = self.grid.append_record(tokens)
        self.assertEqual(record.digit_vector, [-1, -2, -3])

    def test_multiple_records(self):
        self.grid.append_record(Encoder.encode_record(1))
        self.grid.append_record(Encoder.encode_record(2))
        self.grid.append_record(Encoder.encode_record(3))
        self.assertEqual(self.grid.record_count, 3)
        self.assertEqual(self.grid.get_record(0).value_vector, [1])
        self.assertEqual(self.grid.get_record(1).value_vector, [2])
        self.assertEqual(self.grid.get_record(2).value_vector, [3])

    def test_read_at(self):
        self.grid.append_tokens([Token.D1, Token.D2, Token.D3, Token.D4, Token.D5])
        # Read from bit 5 (second token)
        tokens = self.grid.read_at(5, 2)
        self.assertEqual(tokens, [Token.D2, Token.D3])

    def test_read_at_requires_alignment(self):
        with self.assertRaises(ValueError):
            self.grid.read_at(3, 1)  # Not aligned to 5-bit boundary

    def test_pack_unpack_grid(self):
        """Full grid serialization roundtrip."""
        self.grid.append_record(Encoder.encode_record(1, 2))
        self.grid.append_record(Encoder.encode_record(-123, 8175))

        packed, pad = self.grid.pack()
        restored = BinaryGrid.from_packed(packed, pad)

        self.assertEqual(restored.token_count, self.grid.token_count)
        self.assertEqual(restored.record_count, self.grid.record_count)
        self.assertEqual(restored.get_record(0).value_vector, [1, 2])
        self.assertEqual(restored.get_record(1).value_vector, [-123, 8175])

    def test_pack_unpack_empty_grid(self):
        packed, pad = self.grid.pack()
        restored = BinaryGrid.from_packed(packed, pad)
        self.assertEqual(restored.token_count, 0)
        self.assertEqual(restored.record_count, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GEOMETRIC QUERIES
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometricQueries(unittest.TestCase):
    """Verify Hamming distance (address proximity) and Manhattan distance (value proximity)."""

    def test_hamming_same(self):
        self.assertEqual(hamming_distance(0b10101, 0b10101), 0)

    def test_hamming_one_bit(self):
        self.assertEqual(hamming_distance(0b10101, 0b10100), 1)

    def test_hamming_all_different(self):
        self.assertEqual(hamming_distance(0b00000, 0b11111), 5)

    def test_hamming_large_addresses(self):
        """Should work with larger bit-addresses."""
        self.assertEqual(hamming_distance(0xABCD, 0xABCE), 2)  # 0xD ^ 0xE = 0b1101 ^ 0b1110 = 0b0011 → 2 bits

    def test_manhattan_same(self):
        self.assertEqual(manhattan_distance([1, 2, 3], [1, 2, 3]), 0)

    def test_manhattan_simple(self):
        self.assertEqual(manhattan_distance([0, 0], [3, 4]), 7)

    def test_manhattan_negative(self):
        self.assertEqual(manhattan_distance([-1, -2], [1, 2]), 6)

    def test_manhattan_unequal_length_pads_with_zero(self):
        """Shorter vector is zero-padded."""
        self.assertEqual(manhattan_distance([1, 2, 3], [1, 2]), 3)  # |3-0| = 3

    def test_manhattan_empty_both(self):
        self.assertEqual(manhattan_distance([], []), 0)

    def test_manhattan_one_empty(self):
        self.assertEqual(manhattan_distance([1, 2, 3], []), 6)  # 1+2+3

    def test_query_by_manhattan_finds_nearby(self):
        grid = BinaryGrid()
        grid.append_record(Encoder.encode_record(1, 2, 3))
        grid.append_record(Encoder.encode_record(2, 3, 4))
        grid.append_record(Encoder.encode_record(10, 20, 30))
        grid.append_record(Encoder.encode_record(0, 1, 2))

        results = query_by_manhattan(grid, [1, 2, 3], 10)
        values = [r.value_vector for r in results]
        self.assertIn([1, 2, 3], values)
        self.assertIn([2, 3, 4], values)
        self.assertIn([0, 1, 2], values)
        self.assertNotIn([10, 20, 30], values)

    def test_query_by_manhattan_exact_boundary(self):
        """Distance exactly equal to max_distance is excluded (< not <=)."""
        grid = BinaryGrid()
        grid.append_record(Encoder.encode_record(5, 0))  # distance = 5

        results = query_by_manhattan(grid, [0, 0], 5)
        self.assertEqual(len(results), 0)  # distance 5 is NOT < 5

        results = query_by_manhattan(grid, [0, 0], 6)
        self.assertEqual(len(results), 1)

    def test_hamming_shard_routing(self):
        """Find the closest shard by Hamming distance."""
        shards = [0b00000, 0b11111, 0b01010, 0b10101]
        best = query_by_hamming_shard(0b10100, shards)
        self.assertEqual(best, 3)  # 10101 differs by 1 bit from 10100

    def test_hamming_shard_exact_match(self):
        shards = [0b00000, 0b11111, 0b10101]
        best = query_by_hamming_shard(0b11111, shards)
        self.assertEqual(best, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CORRUPTION DETECTION & RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorruptionDetection(unittest.TestCase):
    """Verify bit-flip detection, checksum scanning, and resynchronization."""

    def test_inject_bit_flip_changes_value(self):
        original = [Token.D1, Token.D2, Token.D3]  # 1, 2, 3
        corrupted = inject_bit_flip(original, position=1, bit_index=0)
        # D2 = 00010, flipping MSB → 10010 = N2
        self.assertNotEqual(original, corrupted)
        self.assertEqual(corrupted[1], Token.N2)

    def test_inject_bit_flip_preserves_others(self):
        original = [Token.D1, Token.D2, Token.D3]
        corrupted = inject_bit_flip(original, position=1, bit_index=4)
        self.assertEqual(corrupted[0], original[0])
        self.assertEqual(corrupted[2], original[2])

    def test_scan_for_corruption_clean(self):
        tokens = Encoder.encode_expression([[-1, -2, -3], '*', [-8, -1, -7, -5]])
        with_checksum = append_checksum(tokens)
        results = scan_for_corruption(with_checksum)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)

    def test_scan_for_corruption_detects_flip(self):
        tokens = Encoder.encode_expression([[1], '+', [2]])
        with_checksum = append_checksum(tokens)
        corrupted = inject_bit_flip(with_checksum, position=0, bit_index=1)
        results = scan_for_corruption(corrupted)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)

    def test_scan_multiple_checksums(self):
        """Multiple CHECKSUM segments in a stream."""
        seg1 = append_checksum(Encoder.encode_integer(123))
        seg2 = append_checksum(Encoder.encode_integer(456))
        combined = seg1 + seg2
        results = scan_for_corruption(combined)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.passed for r in results))

    def test_scan_corruption_in_second_segment(self):
        """Corruption in segment 2 shouldn't affect segment 1 check."""
        seg1 = append_checksum(Encoder.encode_integer(123))
        seg2 = append_checksum(Encoder.encode_integer(456))
        # Corrupt segment 2 (inject flip in its data, before its checksum)
        seg2_corrupt = inject_bit_flip(seg2, position=2, bit_index=0)
        combined = seg1 + seg2_corrupt
        results = scan_for_corruption(combined)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].passed, "Segment 1 should still pass")
        self.assertFalse(results[1].passed, "Segment 2 should fail")

    def test_find_next_sync_point_record(self):
        tokens = [
            Token.D1, Token.D2,  # partial data
            Token.RECORD,         # sync point
            Token.D3, Token.END,
        ]
        idx = find_next_sync_point(tokens)
        self.assertEqual(idx, 2)  # Position of RECORD
        self.assertEqual(tokens[idx], Token.RECORD)

    def test_find_next_sync_point_checksum(self):
        tokens = [Token.D1, Token.CHECKSUM, Token.D2]
        idx = find_next_sync_point(tokens)
        self.assertEqual(idx, 1)
        self.assertEqual(tokens[idx], Token.CHECKSUM)

    def test_find_next_sync_point_none(self):
        tokens = [Token.D1, Token.D2, Token.D3, Token.END]
        idx = find_next_sync_point(tokens)
        self.assertIsNone(idx)

    def test_find_next_sync_point_from_offset(self):
        tokens = [Token.RECORD, Token.D1, Token.RECORD, Token.D2]
        idx = find_next_sync_point(tokens, start=1)
        self.assertEqual(idx, 2)  # Second RECORD


# ═══════════════════════════════════════════════════════════════════════════════
# 10. DECIMAL ARITHMETIC (Application Layer)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecimalArithmetic(unittest.TestCase):
    """Verify application-layer decimal arithmetic using (numerator, scale) pairs."""

    def test_from_float_simple(self):
        result = DecimalArithmetic.from_float(0.1)
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.scale, 1)

    def test_from_float_complex(self):
        result = DecimalArithmetic.from_float(1.234)
        self.assertEqual(result.numerator, 1234)
        self.assertEqual(result.scale, 3)

    def test_align_same_scale(self):
        a = ParsedScaledNumber(1, 2)
        b = ParsedScaledNumber(3, 2)
        aa, bb = DecimalArithmetic.align(a, b)
        self.assertEqual(aa.scale, 2)
        self.assertEqual(bb.scale, 2)
        self.assertEqual(aa.numerator, 1)
        self.assertEqual(bb.numerator, 3)

    def test_align_different_scales(self):
        """1/10^1 (0.1) aligned with 2/10^2 (0.02) → 10/10^2 and 2/10^2"""
        a = ParsedScaledNumber(1, 1)   # 0.1
        b = ParsedScaledNumber(2, 2)   # 0.02
        aa, bb = DecimalArithmetic.align(a, b)
        self.assertEqual(aa.scale, 2)
        self.assertEqual(aa.numerator, 10)  # 1 * 10^(2-1)
        self.assertEqual(bb.scale, 2)
        self.assertEqual(bb.numerator, 2)

    def test_add_same_scale(self):
        """0.1 + 0.3 = 0.4: (1,1) + (3,1) = (4,1)"""
        a = ParsedScaledNumber(1, 1)
        b = ParsedScaledNumber(3, 1)
        result = DecimalArithmetic.add(a, b)
        self.assertEqual(result.numerator, 4)
        self.assertEqual(result.scale, 1)
        self.assertAlmostEqual(result.as_float, 0.4)

    def test_add_different_scales(self):
        """0.1 + 0.02 = 0.12: (1,1) + (2,2) = (12,2)"""
        a = ParsedScaledNumber(1, 1)
        b = ParsedScaledNumber(2, 2)
        result = DecimalArithmetic.add(a, b)
        self.assertEqual(result.numerator, 12)
        self.assertEqual(result.scale, 2)
        self.assertAlmostEqual(result.as_float, 0.12)

    def test_subtract(self):
        """0.3 - 0.01 = 0.29: (3,1) - (1,2) = (30-1,2) = (29,2)"""
        a = ParsedScaledNumber(3, 1)
        b = ParsedScaledNumber(1, 2)
        result = DecimalArithmetic.subtract(a, b)
        self.assertEqual(result.numerator, 29)
        self.assertEqual(result.scale, 2)
        self.assertAlmostEqual(result.as_float, 0.29)

    def test_multiply(self):
        """0.1 * 0.02 = 0.002: (1,1) * (2,2) = (2,3)"""
        a = ParsedScaledNumber(1, 1)
        b = ParsedScaledNumber(2, 2)
        result = DecimalArithmetic.multiply(a, b)
        self.assertEqual(result.numerator, 2)
        self.assertEqual(result.scale, 3)
        self.assertAlmostEqual(result.as_float, 0.002)

    def test_database_pattern(self):
        """The full pattern: encode → parse → resolve → arithmetic → write back."""
        # Encode: -1.234 + 5.678
        tokens_a = Encoder.encode_expression([[-1, -2, -3, -4], 'S', [3]])  # -1.234
        tokens_b = Encoder.encode_expression([[5, 6, 7, 8], 'S', [3]])     # 5.678

        # Parse each
        p = Parser()
        p.feed_tokens(tokens_a)
        p.finalize()
        a = resolve_scaled_numbers(p.output)[0]

        p.reset()
        p.feed_tokens(tokens_b)
        p.finalize()
        b = resolve_scaled_numbers(p.output)[0]

        # Application-layer arithmetic
        result = DecimalArithmetic.add(a, b)
        self.assertEqual(result.numerator, 4444)  # -1234 + 5678 = 4444
        self.assertEqual(result.scale, 3)
        self.assertAlmostEqual(result.as_float, 4.444)


# 11. HIGH-LEVEL API (BinaryGridDB)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinaryGridDB(unittest.TestCase):
    """Verify the high-level convenience API."""

    def setUp(self):
        self.db = BinaryGridDB()

    def test_insert_number(self):
        tokens = self.db.insert_number(42)
        self.assertEqual(tokens, [Token.D4, Token.D2, Token.END])
        self.assertEqual(self.db.grid.token_count, 3)

    def test_insert_word(self):
        tokens = self.db.insert_word("HI")
        self.assertIn(Token.START, tokens)
        self.assertIn(Token.END, tokens)

    def test_insert_record(self):
        record = self.db.insert_record(1, 2, 3)
        self.assertEqual(record.value_vector, [1, 2, 3])
        self.assertEqual(self.db.grid.record_count, 1)

    def test_insert_expression(self):
        result = self.db.insert_expression([-1, -2, -3], '*', [-8, -1, -7, -5])
        self.assertEqual(result, 1_005_525)

    def test_insert_expression_simple(self):
        result = self.db.insert_expression([2], '+', [3])
        self.assertEqual(result, 5)

    def test_query_manhattan(self):
        self.db.insert_record(1, 0, 0)
        self.db.insert_record(10, 0, 0)
        self.db.insert_record(2, 1, 0)

        results = self.db.query_manhattan([0, 0, 0], 5)
        vectors = [r.value_vector for r in results]
        self.assertIn([1, 0, 0], vectors)
        self.assertIn([2, 1, 0], vectors)
        self.assertNotIn([10, 0, 0], vectors)

    def test_pack_unpack_db(self):
        self.db.insert_record(1, 2)
        self.db.insert_word("HELLO")
        self.db.insert_number(-5)

        packed, pad = self.db.pack()
        restored = BinaryGridDB.unpack(packed, pad)

        self.assertEqual(restored.grid.token_count, self.db.grid.token_count)
        self.assertEqual(restored.grid.record_count, self.db.grid.record_count)

    def test_stats(self):
        self.db.insert_record(1, 2, 3)
        self.db.insert_record(4, 5, 6)

        stats = self.db.stats()
        self.assertEqual(stats['record_count'], 2)
        self.assertGreater(stats['token_count'], 0)
        self.assertGreater(stats['bit_length'], 0)
        self.assertGreater(stats['byte_length'], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SPEC EXAMPLES (Appendix B) — EXACT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecExamples(unittest.TestCase):
    """Verify every example from Appendix B of the specification."""

    def test_spec_number_123(self):
        """Number 123: [1,2,3] → 00001 00010 00011 11110"""
        tokens = Encoder.encode_integer(123)
        self.assertEqual(
            token_stream_to_binary_string(tokens),
            "00001 00010 00011 11110"
        )

    def test_spec_number_neg123(self):
        """Number -123: [-1,-2,-3] → 10001 10010 10011 11110"""
        tokens = Encoder.encode_integer(-123)
        self.assertEqual(
            token_stream_to_binary_string(tokens),
            "10001 10010 10011 11110"
        )

    def test_spec_word_hi(self):
        """Word 'HI': START [H][I] END → 11111 00111 01000 11110"""
        tokens = Encoder.encode_word("HI")
        self.assertEqual(
            token_stream_to_binary_string(tokens),
            "11111 00111 01000 11110"
        )

    def test_spec_record_1_2(self):
        """Record (1, 2): [1] RECORD [2] RECORD → 00001 11110 11100 00010 11110 11100"""
        r1 = Encoder.encode_record(1)
        r2 = Encoder.encode_record(2)
        combined = r1 + r2
        self.assertEqual(
            token_stream_to_binary_string(combined),
            "00001 11110 11100 00010 11110 11100"
        )

    def test_spec_expression_2_plus_neg3(self):
        """Expression 2 + (-3): [2][+][-3] → 00010 11110 01010 10011 11110"""
        tokens = Encoder.encode_expression([[2], '+', [-3]])
        self.assertEqual(
            token_stream_to_binary_string(tokens),
            "00010 11110 01010 10011 11110"
        )

    def test_spec_scale_neg1_23(self):
        """Scale -1.23: [-1,-2,-3] S [2] → 10001 10010 10011 11110 11011 00010 11110"""
        tokens = Encoder.encode_expression([[-1, -2, -3], 'S', [2]])
        self.assertEqual(
            token_stream_to_binary_string(tokens),
            "10001 10010 10011 11110 11011 00010 11110"
        )

    def test_spec_checksum_block(self):
        """Checksum after data: ...data... 11101 XXXXX (checksum value)"""
        data = Encoder.encode_integer(123) + Encoder.encode_integer(456)
        cs_val = compute_checksum(data)
        with_checksum = append_checksum(data)
        self.assertEqual(with_checksum[-2], Token.CHECKSUM)
        self.assertEqual(int(with_checksum[-1]), cs_val)
        # Verify the CHECKSUM marker is 11101
        self.assertEqual(f'{int(Token.CHECKSUM):05b}', '11101')

    def test_spec_expression_result_encoding(self):
        """-123 * -8175 = 1,005,525 — result should encode as expected."""
        tokens = Encoder.encode_integer(1_005_525)
        self.assertEqual(
            token_stream_to_binary_string(tokens),
            "00001 00000 00000 00101 00101 00010 00101 11110"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 13. EDGE CASES & STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """Adversarial inputs, boundary conditions, and stress scenarios."""

    def test_encode_max_integer_python(self):
        """Should handle Python's arbitrary-precision integers."""
        big = 10**100
        tokens = Encoder.encode_integer(big)
        self.assertEqual(tokens[-1], Token.END)
        # 10**100 = 1 followed by 100 zeros → 101 digits + END = 102 tokens
        self.assertEqual(len(tokens), 102)

    def test_encode_min_integer_python(self):
        big_neg = -(10**100)
        tokens = Encoder.encode_integer(big_neg)
        self.assertEqual(tokens[-1], Token.END)

    def test_word_max_length(self):
        """Encode and decode a very long word."""
        long_word = "A" * 1000
        tokens = Encoder.encode_word(long_word)
        self.assertEqual(tokens[0], Token.START)
        self.assertEqual(tokens[-1], Token.END)
        self.assertEqual(len(tokens), 1002)  # START + 1000*A + END

    def test_empty_expression(self):
        """Empty expression should evaluate to 0."""
        tokens = []
        parser = Parser()
        parser.feed_tokens(tokens)
        parser.finalize()
        result = ArithmeticEvaluator.evaluate_parsed(parser.output)
        # No tokens means no result — actually let me check
        self.assertEqual(result, 0)

    def test_record_with_word_then_number(self):
        """Record containing both a word and a number."""
        tokens = Encoder.encode_record("AGE", 42)
        self.assertEqual(tokens[-1], Token.RECORD)
        # Should contain START (for word) and END (for word) and END (for number)
        self.assertIn(Token.START, tokens)

    def test_multiple_consecutive_ends(self):
        """Multiple END tokens should produce empty numbers."""
        parser = Parser()
        tokens = [Token.END, Token.END, Token.D5, Token.END]
        parser.feed_tokens(tokens)
        parser.finalize()
        # Two empty ENDs followed by NUM(5)
        numbers = [p for p in parser.output if isinstance(p, ParsedNumber)]
        self.assertEqual(len(numbers), 1)
        self.assertEqual(numbers[0].value, 5)

    def test_start_immediately_followed_by_end(self):
        """START END — empty word."""
        parser = Parser()
        tokens = [Token.START, Token.END]
        parser.feed_tokens(tokens)
        parser.finalize()
        words = [p for p in parser.output if isinstance(p, ParsedWord)]
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].text, "")

    def test_grid_with_many_records(self):
        """Insert 1000 records and verify all are stored correctly."""
        grid = BinaryGrid()
        for i in range(1000):
            grid.append_record(Encoder.encode_record(i))
        self.assertEqual(grid.record_count, 1000)
        self.assertEqual(grid.get_record(0).value_vector, [0])
        self.assertEqual(grid.get_record(999).value_vector, [999])

    def test_parser_reset(self):
        """Parser.reset() should clear all state."""
        parser = Parser()
        parser.feed_tokens(Encoder.encode_integer(42))
        parser.feed_tokens(Encoder.encode_word("HI"))
        parser.reset()
        self.assertEqual(parser.state, ParserState.NUM)
        self.assertEqual(parser.accumulator, [])
        self.assertEqual(parser.output, [])
        self.assertEqual(parser.records, [])

    def test_digit_vector_flattening(self):
        """digit_vector should flatten all digits across all numbers in a record."""
        tokens = Encoder.encode_record(12, -34)
        grid = BinaryGrid()
        record = grid.append_record(tokens)
        # 12 → digits [1, 2], -34 → digits [-3, -4]
        self.assertEqual(record.digit_vector, [1, 2, -3, -4])

    def test_negative_zero_not_possible(self):
        """There is no -0 token, and 0 encodes as D0."""
        tokens = Encoder.encode_integer(0)
        self.assertEqual(tokens[0], Token.D0)
        self.assertNotEqual(tokens[0], Token.N1)  # -0 doesn't exist

    def test_digit_token_values_are_distinct(self):
        """Every digit token has a unique value."""
        values = list(NUMERIC_DIGIT_VALUE.values())
        non_none = [v for v in values if v is not None]
        self.assertEqual(len(non_none), len(set(non_none)))

    def test_large_grid_pack_unpack(self):
        """Serialize and deserialize a grid with many records."""
        db = BinaryGridDB()
        for i in range(100):
            db.insert_record(i, i * 2, i * 3)

        packed, pad = db.pack()
        restored = BinaryGridDB.unpack(packed, pad)

        self.assertEqual(restored.grid.record_count, 100)
        self.assertEqual(
            restored.grid.get_record(50).value_vector,
            [50, 100, 150]
        )

    def test_hamming_distance_symmetry(self):
        """Hamming distance is symmetric."""
        for _ in range(100):
            a = hash(str(_)) & 0xFFFF
            b = hash(str(_ + 1)) & 0xFFFF
            self.assertEqual(hamming_distance(a, b), hamming_distance(b, a))

    def test_manhattan_distance_symmetry(self):
        """Manhattan distance is symmetric."""
        self.assertEqual(
            manhattan_distance([1, -2, 3], [-4, 5, -6]),
            manhattan_distance([-4, 5, -6], [1, -2, 3])
        )

    def test_manhattan_distance_triangle_inequality(self):
        """d(a,c) ≤ d(a,b) + d(b,c)"""
        a = [1, 2, 3]
        b = [4, -5, 6]
        c = [-7, 8, 9]
        d_ac = manhattan_distance(a, c)
        d_ab = manhattan_distance(a, b)
        d_bc = manhattan_distance(b, c)
        self.assertLessEqual(d_ac, d_ab + d_bc)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)
