#!/usr/bin/env python3
"""
Binary Grid Database — A Unified 5-Bit Integer Fabric
======================================================
Full implementation of the Binary Grid Database specification (Version 2.0).

A novel database architecture built entirely upon 5-bit binary tokens:
  - Signed integers (-9 to 9)
  - English uppercase letters (A-Z), space, period
  - Arithmetic operators (+, -, *, /, =, ^, S for scale)
  - Control codes (START, END, RECORD, CHECKSUM)
  - Shunting-yard arithmetic evaluator
  - Modulo-32 checksum integrity
  - Hamming distance (address proximity) and Manhattan distance (value proximity) queries
  - 5-bit ↔ 8-bit serialization

Author: Claude (based on user specification)
Date: 2026-06-21
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Union, Dict, Set


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TOKEN DEFINITIONS — The Unified 5‑Bit Lexicon
# ═══════════════════════════════════════════════════════════════════════════════

class Token(IntEnum):
    """All 32 five-bit codes. Interpretation is context-dependent."""
    # --- Digits (positive: 0x00–0x09) ---
    D0  = 0b00000
    D1  = 0b00001
    D2  = 0b00010
    D3  = 0b00011
    D4  = 0b00100
    D5  = 0b00101
    D6  = 0b00110
    D7  = 0b00111
    D8  = 0b01000
    D9  = 0b01001

    # --- Operators / Letters 0x0A–0x14 ---
    T_PLUS   = 0b01010  # + in NUM, K in WORD
    T_MINUS  = 0b01011  # - in NUM, L in WORD
    T_MUL    = 0b01100  # * in NUM, M in WORD
    T_DIV    = 0b01101  # / in NUM, N in WORD
    T_EQ     = 0b01110  # = in NUM, O in WORD
    T_LPAREN = 0b01111  # ( in NUM, P in WORD
    T_RPAREN = 0b10000  # ) in NUM, Q in WORD

    # --- Negative digits 0x11–0x19 ---
    N1 = 0b10001  # -1 in NUM, R in WORD
    N2 = 0b10010  # -2 in NUM, S in WORD
    N3 = 0b10011  # -3 in NUM, T in WORD
    N4 = 0b10100  # -4 in NUM, U in WORD
    N5 = 0b10101  # -5 in NUM, V in WORD
    N6 = 0b10110  # -6 in NUM, W in WORD
    N7 = 0b10111  # -7 in NUM, X in WORD
    N8 = 0b11000  # -8 in NUM, Y in WORD
    N9 = 0b11001  # -9 in NUM, Z in WORD

    # --- Extended operators / punctuation 0x1A–0x1B ---
    T_POW   = 0b11010  # ^ in NUM, SPACE in WORD
    T_SCALE = 0b11011  # S in NUM, . in WORD

    # --- Control codes 0x1C–0x1F ---
    RECORD   = 0b11100
    CHECKSUM = 0b11101
    END      = 0b11110
    START    = 0b11111


# ── Mapping tables ───────────────────────────────────────────────────────────

# Numeric context: token → digit value (None for non-digits)
NUMERIC_DIGIT_VALUE: Dict[Token, Optional[int]] = {
    Token.D0: 0,  Token.D1: 1,  Token.D2: 2,  Token.D3: 3,  Token.D4: 4,
    Token.D5: 5,  Token.D6: 6,  Token.D7: 7,  Token.D8: 8,  Token.D9: 9,
    Token.N1: -1, Token.N2: -2, Token.N3: -3, Token.N4: -4, Token.N5: -5,
    Token.N6: -6, Token.N7: -7, Token.N8: -8, Token.N9: -9,
}

# Numeric context: digit value → token
DIGIT_TO_TOKEN: Dict[int, Token] = {v: k for k, v in NUMERIC_DIGIT_VALUE.items() if v is not None}

# Numeric context: arithmetic operator tokens (not digits, not controls)
NUMERIC_OPERATORS: Set[Token] = {
    Token.T_PLUS, Token.T_MINUS, Token.T_MUL, Token.T_DIV,
    Token.T_EQ, Token.T_LPAREN, Token.T_RPAREN, Token.T_POW,
}

# Numeric context: storage annotations (not arithmetic operators)
# S = "Scale" — annotates the preceding integer with N implied decimal places.
# The database stores pure integers; S is metadata for the application layer.
NUMERIC_ANNOTATIONS: Set[Token] = {Token.T_SCALE}

# Operator token → symbol (for display / shunting-yard)
OPERATOR_SYMBOL: Dict[Token, str] = {
    Token.T_PLUS: '+', Token.T_MINUS: '-', Token.T_MUL: '*', Token.T_DIV: '/',
    Token.T_EQ: '=', Token.T_LPAREN: '(', Token.T_RPAREN: ')',
    Token.T_POW: '^', Token.T_SCALE: 'S',
}

SYMBOL_TO_OPERATOR: Dict[str, Token] = {v: k for k, v in OPERATOR_SYMBOL.items()}

# Word context: token → character
WORD_CHAR: Dict[Token, str] = {
    Token.D0: 'A', Token.D1: 'B', Token.D2: 'C', Token.D3: 'D',
    Token.D4: 'E', Token.D5: 'F', Token.D6: 'G', Token.D7: 'H',
    Token.D8: 'I', Token.D9: 'J',
    Token.T_PLUS: 'K', Token.T_MINUS: 'L', Token.T_MUL: 'M', Token.T_DIV: 'N',
    Token.T_EQ: 'O', Token.T_LPAREN: 'P', Token.T_RPAREN: 'Q',
    Token.N1: 'R', Token.N2: 'S', Token.N3: 'T', Token.N4: 'U',
    Token.N5: 'V', Token.N6: 'W', Token.N7: 'X', Token.N8: 'Y', Token.N9: 'Z',
    Token.T_POW: ' ',   # SPACE
    Token.T_SCALE: '.',  # PERIOD
}

CHAR_TO_WORD_TOKEN: Dict[str, Token] = {v: k for k, v in WORD_CHAR.items()}

# ── SPECIAL context: lowercase letters & special characters ──────────────────
# Triggered by START-in-WORD. Uses same 28 slots (00000-11011) but remapped.
# Controls (11100-11111) retain their meaning across all contexts.

SPECIAL_CHAR: Dict[Token, str] = {
    # Lowercase a-z (same positional order as uppercase)
    Token.D0: 'a', Token.D1: 'b', Token.D2: 'c', Token.D3: 'd', Token.D4: 'e',
    Token.D5: 'f', Token.D6: 'g', Token.D7: 'h', Token.D8: 'i', Token.D9: 'j',
    Token.T_PLUS: 'k', Token.T_MINUS: 'l', Token.T_MUL: 'm', Token.T_DIV: 'n',
    Token.T_EQ: 'o', Token.T_LPAREN: 'p', Token.T_RPAREN: 'q',
    Token.N1: 'r', Token.N2: 's', Token.N3: 't', Token.N4: 'u',
    Token.N5: 'v', Token.N6: 'w', Token.N7: 'x', Token.N8: 'y', Token.N9: 'z',
    # Special characters
    Token.T_POW: '@',        # was SPACE in WORD
    Token.T_SCALE: '-',      # was . in WORD
}

CHAR_TO_SPECIAL_TOKEN: Dict[str, Token] = {v: k for k, v in SPECIAL_CHAR.items()}

# ── SPECIAL2 context: extended special characters ──────────────────────────
# Triggered by START-in-SPECIAL.  28 additional slots for punctuation.

SPECIAL2_CHAR: Dict[Token, str] = {
    Token.D0: '!',  Token.D1: '"',  Token.D2: '#',  Token.D3: '$',
    Token.D4: '%',  Token.D5: '&',  Token.D6: "'",  Token.D7: '(',
    Token.D8: ')',  Token.D9: '*',
    Token.T_PLUS: '+',   Token.T_MINUS: ',',  Token.T_MUL: '/',
    Token.T_DIV: ':',    Token.T_EQ: ';',     Token.T_LPAREN: '<',
    Token.T_RPAREN: '=', Token.N1: '>',       Token.N2: '?',
    Token.N3: '[',       Token.N4: '\\',      Token.N5: ']',
    Token.N6: '^',       Token.N7: '_',       Token.N8: '`',
    Token.N9: '{',       Token.T_POW: '|',    Token.T_SCALE: '}',
}

CHAR_TO_SPECIAL2_TOKEN: Dict[str, Token] = {v: k for k, v in SPECIAL2_CHAR.items()}

# Control tokens
CONTROL_TOKENS: Set[Token] = {Token.START, Token.END, Token.RECORD, Token.CHECKSUM}

# All tokens that are considered "digits" in NUM context
DIGIT_TOKENS: Set[Token] = set(NUMERIC_DIGIT_VALUE.keys())

# Display names for tokens
TOKEN_NAME: Dict[Token, str] = {
    Token.D0: '0', Token.D1: '1', Token.D2: '2', Token.D3: '3', Token.D4: '4',
    Token.D5: '5', Token.D6: '6', Token.D7: '7', Token.D8: '8', Token.D9: '9',
    Token.N1: '-1', Token.N2: '-2', Token.N3: '-3', Token.N4: '-4', Token.N5: '-5',
    Token.N6: '-6', Token.N7: '-7', Token.N8: '-8', Token.N9: '-9',
    Token.T_PLUS: '+', Token.T_MINUS: '-', Token.T_MUL: '*', Token.T_DIV: '/',
    Token.T_EQ: '=', Token.T_LPAREN: '(', Token.T_RPAREN: ')',
    Token.T_POW: '^', Token.T_SCALE: 'S',
    Token.RECORD: 'RECORD', Token.CHECKSUM: 'CHECKSUM',
    Token.END: 'END', Token.START: 'START',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARSER STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class ParserState(IntEnum):
    NUM = 0
    WORD = 1
    SPECIAL = 2   # START-in-WORD: lowercase a-z, @, -
    SPECIAL2 = 3  # START-in-SPECIAL: ! " # $ % & ' ( ) * + , / : ; < = > ? [ \ ] ^ _ ` { | }


@dataclass
class ParsedNumber:
    """A parsed multi-digit signed number."""
    digits: List[int]  # Signed digit values, e.g., [-1, -2, -3]
    value: int         # Computed integer value

    def __repr__(self):
        return f"NUM({self.value})"


@dataclass
class ParsedScaledNumber:
    """A decimal number stored as (integer numerator, scale exponent).

    Represents: numerator / 10^scale
    Example: -1234 S 3 → numerator=-1234, scale=3 → -1.234

    The database stores only integers. S is a storage annotation that tells
    the application layer where the implied decimal point goes.
    This is exactly how financial systems store currency (cents) and how
    scientific systems store measurements (integer + unit prefix).
    """
    numerator: int
    scale: int        # Number of decimal places (non-negative)

    @property
    def as_float(self) -> float:
        """Application-layer convenience: interpret as a float."""
        return self.numerator / (10 ** self.scale)

    def __repr__(self):
        return f"SCALED({self.numerator} / 10^{self.scale} = {self.as_float})"


@dataclass
class ParsedWord:
    """A parsed word (string)."""
    characters: List[str]
    text: str

    def __repr__(self):
        return f"WORD('{self.text}')"


@dataclass
class ParsedOperator:
    """A parsed arithmetic operator."""
    token: Token
    symbol: str

    def __repr__(self):
        return f"OP({self.symbol})"


@dataclass
class ChecksumResult:
    """Result of a checksum verification."""
    expected: int
    computed: int
    passed: bool

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"CHECKSUM({status}: expected={self.expected}, computed={self.computed})"


# Union type for parsed tokens
ParsedToken = Union[ParsedNumber, ParsedScaledNumber, ParsedWord, ParsedOperator, Token, ChecksumResult]


@dataclass
class Record:
    """A logical record (tuple) — a sequence of parsed tokens terminated by RECORD."""
    tokens: List[ParsedToken]
    bit_offset: int  # Starting bit offset in the grid

    def __repr__(self):
        return f"Record({self.tokens})"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENCODER — Converts high-level values into 5‑bit token streams
# ═══════════════════════════════════════════════════════════════════════════════

class Encoder:
    """Encodes numbers, words, expressions, and records into 5-bit token streams."""

    @staticmethod
    def encode_integer(value: int) -> List[Token]:
        """Encode a signed integer as signed-digit tokens + END.

        Example: 123  → [D1, D2, D3, END]
        Example: -123 → [N1, N2, N3, END]
        Example: 0    → [D0, END]

        For negative numbers, each digit carries its own sign.
        Zero digits in a negative number (e.g., -105) use D0
        since there is no negative-zero concept.
        """
        if value == 0:
            return [Token.D0, Token.END]

        sign = 1 if value >= 0 else -1
        digits_str = str(abs(value))

        tokens = []
        for ch in digits_str:
            d = int(ch)
            if d == 0:
                # Zero is always D0 — no -0 concept
                tokens.append(Token.D0)
            else:
                prefix = 'N' if sign < 0 else 'D'
                tokens.append(Token[f'{prefix}{d}'])
        tokens.append(Token.END)
        return tokens

    @staticmethod
    def encode_signed_digits(digits: List[int]) -> List[Token]:
        """Encode a list of signed digit values (without trailing END)."""
        tokens = []
        for d in digits:
            tokens.append(DIGIT_TO_TOKEN[d])
        return tokens

    @staticmethod
    def encode_number_from_digits(digits: List[int]) -> List[Token]:
        """Encode signed digits + END."""
        return Encoder.encode_signed_digits(digits) + [Token.END]

    @staticmethod
    def encode_word(text: str) -> List[Token]:
        """Encode a word as START + letter tokens + END.
        Handles mixed WORD/SPECIAL contexts via START-in-WORD switching.

        Example: "HI" → [START, H, I, END] = [11111, 00111, 01000, 11110]
        Example: "hi" → [START, START, h, i, END, END]
        Example: "Hi@there" → START H START i @ t h e r e END END
        """
        tokens = [Token.START]
        depth = 0  # 0=WORD, 1=SPECIAL, 2=SPECIAL2
        pop = lambda target: [Token.END] * (depth - target)  # pop to target depth

        for ch in text:
            if ch.isdigit():
                tokens.extend(pop(0))  # Pop all the way to WORD
                depth = 0
                tokens.append(Token.END)       # WORD → NUM
                tokens.append(DIGIT_TO_TOKEN[int(ch)])
                tokens.append(Token.START)     # NUM → WORD
                continue

            if ch in CHAR_TO_WORD_TOKEN:
                tokens.extend(pop(0)); depth = 0
                tokens.append(CHAR_TO_WORD_TOKEN[ch])
                continue

            if ch in CHAR_TO_SPECIAL_TOKEN:
                if depth > 1: tokens.append(Token.END); depth = 1  # Pop SPECIAL2→SPECIAL
                elif depth < 1: tokens.append(Token.START); depth = 1
                tokens.append(CHAR_TO_SPECIAL_TOKEN[ch])
                continue

            if ch in CHAR_TO_SPECIAL2_TOKEN:
                if depth < 2:
                    if depth < 1: tokens.append(Token.START); depth = 1
                    tokens.append(Token.START); depth = 2
                tokens.append(CHAR_TO_SPECIAL2_TOKEN[ch])
                continue

            if ch.upper() in CHAR_TO_WORD_TOKEN:
                tokens.extend(pop(0)); depth = 0
                tokens.append(CHAR_TO_WORD_TOKEN[ch.upper()])
                continue

            raise ValueError(f"Character '{ch}' cannot be encoded in any context")

        tokens.extend(pop(0))  # Pop all contexts
        tokens.append(Token.END)  # WORD → NUM
        return tokens

    @staticmethod
    def encode_operator(symbol: str) -> Token:
        """Encode an operator symbol to its token."""
        if symbol not in SYMBOL_TO_OPERATOR:
            raise ValueError(f"Unknown operator: '{symbol}'")
        return SYMBOL_TO_OPERATOR[symbol]

    @staticmethod
    def encode_expression(tokens: List[Union[int, str, List[int]]]) -> List[Token]:
        """Encode an expression from a list of values, operators, and digit-lists.

        Each element can be:
          - int: a number (e.g., 123, -5)
          - str: an operator (e.g., '+', '*', 'S')
          - List[int]: signed digits (e.g., [-1, -2, -3])
        """
        result = []
        for item in tokens:
            if isinstance(item, int):
                result.extend(Encoder.encode_integer(item))
            elif isinstance(item, list):
                result.extend(Encoder.encode_number_from_digits(item))
            elif isinstance(item, str):
                result.append(Encoder.encode_operator(item))
            else:
                raise ValueError(f"Cannot encode: {item}")
        return result

    @staticmethod
    def encode_record(*values: Union[int, str, List[Token]]) -> List[Token]:
        """Encode values into a record terminated by RECORD.

        Each value can be an integer, a string, or a pre-encoded token list.
        """
        tokens = []
        for val in values:
            if isinstance(val, int):
                tokens.extend(Encoder.encode_integer(val))
            elif isinstance(val, str):
                tokens.extend(Encoder.encode_word(val))
            elif isinstance(val, list):
                tokens.extend(val)
            else:
                raise ValueError(f"Cannot encode record value: {val}")
        tokens.append(Token.RECORD)
        return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DECODER / PARSER — Finite state machine
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    """Finite-state machine that parses a stream of 5-bit tokens.

    States: NUM (default), WORD
    Transitions driven by START, END, RECORD, CHECKSUM control tokens.
    """

    def __init__(self):
        self.state = ParserState.NUM
        self.accumulator: List[int] = []      # Signed digit values or character strings
        self.output: List[ParsedToken] = []   # Parsed tokens emitted so far
        self.records: List[Record] = []       # Completed records
        self.current_record_start: int = 0    # Token index where current record began
        self.token_count: int = 0             # Total tokens processed
        self.last_checksum_index: int = 0     # Token index after last CHECKSUM

    def reset(self):
        """Reset parser state."""
        self.state = ParserState.NUM
        self.accumulator = []
        self.output = []
        self.records = []
        self.current_record_start = 0
        self.token_count = 0
        self.last_checksum_index = 0

    def _finalize_number(self):
        """Convert accumulated signed digits into a ParsedNumber and emit it."""
        if not self.accumulator:
            return
        digits = list(self.accumulator)
        # Compute value: d1*10^(n-1) + d2*10^(n-2) + ... + dn*10^0
        value = 0
        for d in digits:
            sign = 1 if d >= 0 else -1
            value = value * 10 + d  # d already carries sign per spec
        # Actually, re-read the spec: "Value = d₁ * 10^(n-1) + d₂ * 10^(n-2) + ... + dₙ * 10^0"
        # Where d₁ is the signed digit. So -123 with digits [-1, -2, -3]:
        # -1*100 + -2*10 + -3 = -100 + -20 + -3 = -123 ✓
        value = 0
        n = len(digits)
        for i, d in enumerate(digits):
            value += d * (10 ** (n - 1 - i))

        parsed = ParsedNumber(digits=digits, value=value)
        self.output.append(parsed)
        self.accumulator = []

    def _finalize_word(self):
        """Convert accumulated characters into a ParsedWord and emit it.
        Emits even an empty word (START immediately followed by END → "").
        """
        chars = []
        for t in self.accumulator:
            tok = Token(t)
            chars.append(WORD_CHAR[tok])
        text = ''.join(chars)
        parsed = ParsedWord(characters=chars, text=text)
        self.output.append(parsed)
        self.accumulator = []

    def _finalize_special(self):
        """Convert accumulated SPECIAL tokens into a ParsedWord and emit it.
        Uses SPECIAL_CHAR mapping (lowercase + special chars)."""
        chars = []
        for t in self.accumulator:
            tok = Token(t)
            chars.append(SPECIAL_CHAR[tok])
        text = ''.join(chars)
        parsed = ParsedWord(characters=chars, text=text)
        self.output.append(parsed)
        self.accumulator = []

    def _finalize_special2(self):
        """Convert accumulated SPECIAL2 tokens into a ParsedWord and emit it."""
        chars = []
        for t in self.accumulator:
            tok = Token(t)
            chars.append(SPECIAL2_CHAR[tok])
        text = ''.join(chars)
        parsed = ParsedWord(characters=chars, text=text)
        self.output.append(parsed)
        self.accumulator = []

    def _emit_record(self):
        """Emit a RECORD boundary, grouping tokens since the last RECORD."""
        # Find tokens emitted since last record start
        record_tokens = self.output[self.current_record_start:]
        record = Record(tokens=list(record_tokens), bit_offset=self.current_record_start * 5)
        self.records.append(record)
        self.output.append(Token.RECORD)
        self.current_record_start = len(self.output)

    def feed(self, token: Token) -> Optional[ParsedToken]:
        """Feed a single 5-bit token into the parser. Returns a parsed token if one was emitted."""
        self.token_count += 1
        emitted = None

        if self.state == ParserState.NUM:
            if token == Token.START:
                # Finalize any pending number
                self._finalize_number()
                self.state = ParserState.WORD

            elif token == Token.END:
                self._finalize_number()
                emitted = Token.END

            elif token == Token.RECORD:
                self._finalize_number()
                self._emit_record()
                emitted = Token.RECORD

            elif token == Token.CHECKSUM:
                self._finalize_number()
                # CHECKSUM handling is done externally (we need to read the next 5 bits)
                emitted = Token.CHECKSUM

            elif token in DIGIT_TOKENS:
                self.accumulator.append(NUMERIC_DIGIT_VALUE[token])

            elif token in NUMERIC_OPERATORS:
                self._finalize_number()
                op = ParsedOperator(token=token, symbol=OPERATOR_SYMBOL[token])
                self.output.append(op)
                emitted = op

            elif token in NUMERIC_ANNOTATIONS:
                # Storage annotations (e.g., S for Scale).
                # Emitted as an operator token in the stream, but NOT an arithmetic operator.
                # Post-processing pairs NUM S NUM → ParsedScaledNumber.
                self._finalize_number()
                op = ParsedOperator(token=token, symbol=OPERATOR_SYMBOL[token])
                self.output.append(op)
                emitted = op

            else:
                raise ValueError(f"Unexpected token {token.name} in NUM state")

        elif self.state == ParserState.WORD:
            if token == Token.END:
                self._finalize_word()
                self.state = ParserState.NUM
                emitted = Token.END

            elif token == Token.RECORD:
                self._finalize_word()
                self.state = ParserState.NUM
                self._emit_record()
                emitted = Token.RECORD

            elif token == Token.START:
                # START-in-WORD: enter SPECIAL context (lowercase + special chars)
                self._finalize_word()
                self.state = ParserState.SPECIAL

            elif token == Token.CHECKSUM:
                self._finalize_word()
                self.state = ParserState.NUM
                emitted = Token.CHECKSUM

            elif token in WORD_CHAR:
                self.accumulator.append(int(token))

            else:
                if token in WORD_CHAR:
                    self.accumulator.append(int(token))
                else:
                    raise ValueError(f"Unexpected token {token.name} in WORD state")

        elif self.state == ParserState.SPECIAL:
            if token == Token.END:
                self._finalize_special()
                self.state = ParserState.WORD  # Pop back to WORD
                emitted = Token.END

            elif token == Token.RECORD:
                self._finalize_special()
                self.state = ParserState.NUM  # Pop all the way to NUM
                self._emit_record()
                emitted = Token.RECORD

            elif token == Token.CHECKSUM:
                self._finalize_special()
                self.state = ParserState.NUM
                emitted = Token.CHECKSUM

            elif token == Token.START:
                # START-in-SPECIAL: enter SPECIAL2 (extended punctuation)
                self._finalize_special()
                self.state = ParserState.SPECIAL2

            elif token in SPECIAL_CHAR:
                self.accumulator.append(int(token))

            else:
                raise ValueError(f"Unexpected token {token.name} in SPECIAL state")

        elif self.state == ParserState.SPECIAL2:
            if token == Token.END:
                self._finalize_special2()
                self.state = ParserState.SPECIAL
                emitted = Token.END
            elif token == Token.RECORD:
                self._finalize_special2()
                self.state = ParserState.NUM
                self._emit_record()
                emitted = Token.RECORD
            elif token == Token.CHECKSUM:
                self._finalize_special2()
                self.state = ParserState.NUM
                emitted = Token.CHECKSUM
            elif token == Token.START:
                pass
            elif token in SPECIAL2_CHAR:
                self.accumulator.append(int(token))
            else:
                raise ValueError(f"Unexpected token {token.name} in SPECIAL2 state")

        return emitted

    def feed_tokens(self, tokens: List[Token]) -> List[ParsedToken]:
        """Feed a list of tokens. Returns all emitted parsed tokens."""
        emitted = []
        for t in tokens:
            result = self.feed(t)
            if result is not None:
                emitted.append(result)
        return emitted

    def finalize(self):
        """Finalize any pending accumulator."""
        if self.state == ParserState.NUM:
            self._finalize_number()
        elif self.state == ParserState.WORD:
            self._finalize_word()
        elif self.state == ParserState.SPECIAL:
            self._finalize_special()
        elif self.state == ParserState.SPECIAL2:
            self._finalize_special2()

    def reassemble(self):
        """Reassemble fragmented words: merge consecutive WORD tokens, drop empties."""
        merged = []
        pending = ''
        for p in self.output:
            if isinstance(p, ParsedWord):
                pending += p.text
            else:
                if pending:
                    merged.append(ParsedWord(characters=list(pending), text=pending))
                    pending = ''
                merged.append(p)
        if pending:
            merged.append(ParsedWord(characters=list(pending), text=pending))
        self.output = merged


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CHECKSUM — Modulo-32 integrity
# ═══════════════════════════════════════════════════════════════════════════════

def compute_checksum(tokens: List[Token]) -> int:
    """Compute modulo-32 checksum: sum of all token integer values % 32."""
    total = sum(int(t) for t in tokens)
    return total % 32


def verify_checksum(tokens: List[Token], expected: int) -> ChecksumResult:
    """Verify that a checksum value matches the computed checksum of tokens."""
    computed = compute_checksum(tokens)
    return ChecksumResult(expected=expected, computed=computed, passed=(computed == expected))


def append_checksum(tokens: List[Token]) -> List[Token]:
    """Append a CHECKSUM token + checksum value to a token list."""
    cs_value = compute_checksum(tokens)
    result = list(tokens)
    result.append(Token.CHECKSUM)
    # The checksum value is encoded as a 5-bit value directly
    # We need to find a token whose integer value equals cs_value
    cs_token = _token_for_value(cs_value)
    result.append(cs_token)
    return result


def _token_for_value(value: int) -> Token:
    """Find a token whose integer value equals the given value (for checksum payload)."""
    for t in Token:
        if int(t) == value:
            return t
    raise ValueError(f"No token with value {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SERIALIZATION — 5‑bit ↔ 8‑bit packing
# ═══════════════════════════════════════════════════════════════════════════════

def pack_to_bytes(tokens: List[Token]) -> Tuple[bytes, int]:
    """Pack a list of 5-bit tokens into a byte array.

    Returns (bytes, pad_length) where pad_length is the number of zero bits
    padded to reach a byte boundary (0-7), for lossless unpacking.
    """
    # Build bit string
    bits = []
    for t in tokens:
        val = int(t)
        for i in range(4, -1, -1):
            bits.append((val >> i) & 1)

    # Pad to byte boundary
    pad_length = (8 - (len(bits) % 8)) % 8
    bits.extend([0] * pad_length)

    # Pack into bytes
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bits[i + j]
        byte_array.append(byte_val)

    return bytes(byte_array), pad_length


def unpack_from_bytes(data: bytes, pad_length: int = 0, num_tokens: Optional[int] = None) -> List[Token]:
    """Unpack bytes back into 5-bit tokens.

    Args:
        data: The byte array to unpack.
        pad_length: Number of zero bits padded at the end (from pack_to_bytes).
        num_tokens: If specified, unpack exactly this many tokens.
                    Otherwise, unpack all complete tokens.

    Returns:
        List of Token values.
    """
    # Convert bytes to bit stream
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    # Remove padding
    if pad_length > 0:
        bits = bits[:-pad_length]

    # Extract 5-bit tokens
    tokens = []
    i = 0
    while i + 5 <= len(bits):
        if num_tokens is not None and len(tokens) >= num_tokens:
            break
        val = 0
        for j in range(5):
            val = (val << 1) | bits[i + j]
        tokens.append(Token(val))
        i += 5

    return tokens


def token_stream_to_binary_string(tokens: List[Token]) -> str:
    """Convert a token list to a space-separated binary string (for debugging)."""
    return ' '.join(f'{int(t):05b}' for t in tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ARITHMETIC EVALUATOR — Shunting‑Yard + RPN
# ═══════════════════════════════════════════════════════════════════════════════

class ArithmeticEvaluator:
    """Evaluates arithmetic expressions encoded as 5-bit tokens.

    Uses the Shunting-Yard algorithm to convert infix to RPN,
    then evaluates with a stack machine operating on pure integers.

    The Scale token (S) is NOT an arithmetic operator — it is a storage
    annotation.  NUM S NUM patterns are resolved to ParsedScaledNumber
    by resolve_scaled_numbers() before arithmetic evaluation, or left
    as-is when the expression is purely about storage (e.g., a record).
    """

    # Precedence (higher = binds tighter)
    PRECEDENCE = {
        Token.T_EQ: 1,
        Token.T_PLUS: 2,
        Token.T_MINUS: 2,
        Token.T_MUL: 3,
        Token.T_DIV: 3,
        Token.T_POW: 4,
    }

    # Right-associative operators
    RIGHT_ASSOCIATIVE = {Token.T_POW, Token.T_EQ}

    @staticmethod
    def _is_operator(token: Token) -> bool:
        return token in NUMERIC_OPERATORS

    @staticmethod
    def _is_annotation(token: Token) -> bool:
        return token in NUMERIC_ANNOTATIONS

    @staticmethod
    def _extract_expression_tokens(parsed_tokens: List[ParsedToken]) -> List[Union[ParsedNumber, ParsedOperator]]:
        """Extract numbers and arithmetic operators, skipping annotations (S)."""
        expr = []
        for pt in parsed_tokens:
            if isinstance(pt, ParsedNumber) or isinstance(pt, ParsedScaledNumber):
                expr.append(pt)
            elif isinstance(pt, ParsedOperator):
                if pt.token not in NUMERIC_ANNOTATIONS:
                    expr.append(pt)
                # Skip annotation tokens — they're resolved separately
            # Skip END, RECORD, etc.
        return expr

    @classmethod
    def evaluate_parsed(cls, parsed_tokens: List[ParsedToken]) -> int:
        """Evaluate an expression from parsed tokens. Returns the integer result.

        First resolves NUM S NUM → ParsedScaledNumber, then evaluates
        arithmetic on their integer numerators.  The S annotation is
        consumed during resolution — it does not appear as an operator
        in the shunting-yard stage.

        The expression should be a sequence of NUM tokens and OP tokens.
        """
        # Step 1: Resolve scaled numbers (consume S annotations)
        resolved = resolve_scaled_numbers(parsed_tokens)

        # Step 2: Extract numbers and arithmetic operators only
        expr_tokens = cls._extract_expression_tokens(resolved)

        if not expr_tokens:
            return 0

        # If it's just a single number, return its integer value
        if len(expr_tokens) == 1:
            item = expr_tokens[0]
            if isinstance(item, ParsedNumber):
                return item.value
            elif isinstance(item, ParsedScaledNumber):
                return item.numerator

        # Shunting-yard: infix → RPN
        output_queue: List[Union[ParsedNumber, ParsedScaledNumber, ParsedOperator]] = []
        operator_stack: List[Token] = []

        for item in expr_tokens:
            if isinstance(item, (ParsedNumber, ParsedScaledNumber)):
                output_queue.append(item)

            elif isinstance(item, ParsedOperator):
                token = item.token
                if token == Token.T_LPAREN:
                    operator_stack.append(token)
                elif token == Token.T_RPAREN:
                    while operator_stack and operator_stack[-1] != Token.T_LPAREN:
                        output_queue.append(ParsedOperator(
                            token=operator_stack[-1],
                            symbol=OPERATOR_SYMBOL[operator_stack[-1]]
                        ))
                        operator_stack.pop()
                    if operator_stack and operator_stack[-1] == Token.T_LPAREN:
                        operator_stack.pop()
                    else:
                        raise ValueError("Mismatched parentheses")
                else:
                    # Arithmetic operator
                    while (operator_stack and operator_stack[-1] != Token.T_LPAREN and
                           (cls.PRECEDENCE.get(operator_stack[-1], 0) > cls.PRECEDENCE.get(token, 0) or
                            (cls.PRECEDENCE.get(operator_stack[-1], 0) == cls.PRECEDENCE.get(token, 0) and
                             token not in cls.RIGHT_ASSOCIATIVE))):
                        output_queue.append(ParsedOperator(
                            token=operator_stack[-1],
                            symbol=OPERATOR_SYMBOL[operator_stack[-1]]
                        ))
                        operator_stack.pop()
                    operator_stack.append(token)

        # Pop remaining operators
        while operator_stack:
            top = operator_stack.pop()
            if top == Token.T_LPAREN:
                raise ValueError("Mismatched parentheses")
            output_queue.append(ParsedOperator(token=top, symbol=OPERATOR_SYMBOL[top]))

        # Evaluate RPN — operate on integer numerators of scaled numbers
        stack: List[int] = []
        for item in output_queue:
            if isinstance(item, ParsedNumber):
                stack.append(item.value)
            elif isinstance(item, ParsedScaledNumber):
                stack.append(item.numerator)
            elif isinstance(item, ParsedOperator):
                if len(stack) < 2:
                    raise ValueError(f"Operator '{item.symbol}' requires two operands")
                b = stack.pop()
                a = stack.pop()

                if item.token == Token.T_PLUS:
                    stack.append(a + b)
                elif item.token == Token.T_MINUS:
                    stack.append(a - b)
                elif item.token == Token.T_MUL:
                    stack.append(a * b)
                elif item.token == Token.T_DIV:
                    if b == 0:
                        raise ZeroDivisionError("Division by zero")
                    stack.append(a // b)
                elif item.token == Token.T_POW:
                    stack.append(a ** b)
                elif item.token == Token.T_EQ:
                    stack.append(1 if a == b else 0)
                else:
                    raise ValueError(f"Unknown operator: {item.symbol}")

        if not stack:
            return 0
        return stack[-1]


def resolve_scaled_numbers(parsed_tokens: List[ParsedToken]) -> List[ParsedToken]:
    """Post-process parsed tokens: pair NUM S NUM → ParsedScaledNumber.

    The S token is a storage annotation, not an arithmetic operator.
    It travels alongside the preceding number in the token stream.
    This function consumes S and the following number, replacing the
    three-token sequence (NUM, OP('S'), NUM) with a single ParsedScaledNumber.

    Example:
        [NUM(-1234), OP('S'), NUM(3)] → [SCALED(-1234 / 10^3 = -1.234)]
    """
    result = []
    i = 0
    while i < len(parsed_tokens):
        token = parsed_tokens[i]

        # Look for pattern: ParsedNumber followed by S annotation followed by ParsedNumber
        if (isinstance(token, ParsedNumber) and
            i + 2 < len(parsed_tokens) and
            isinstance(parsed_tokens[i + 1], ParsedOperator) and
            parsed_tokens[i + 1].token == Token.T_SCALE and
            isinstance(parsed_tokens[i + 2], ParsedNumber)):

            numerator = token.value
            scale = parsed_tokens[i + 2].value
            if scale < 0:
                raise ValueError(f"Scale exponent must be non-negative, got {scale}")
            result.append(ParsedScaledNumber(numerator=numerator, scale=scale))
            i += 3  # Consume all three tokens
        else:
            result.append(token)
            i += 1

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7b. DECIMAL ARITHMETIC — Application-layer scale alignment
# ═══════════════════════════════════════════════════════════════════════════════

class DecimalArithmetic:
    """Application-layer decimal arithmetic using the (numerator, scale) model.

    The database stores pure integers.  This class provides scale-aware
    operations for the application layer — exactly how financial systems
    handle currency (store cents, display dollars) and scientific systems
    handle measurements (store integer + unit prefix).

    All operations return ParsedScaledNumber results suitable for writing
    back to the grid.
    """

    @staticmethod
    def align(a: ParsedScaledNumber, b: ParsedScaledNumber) -> Tuple[ParsedScaledNumber, ParsedScaledNumber]:
        """Align two scaled numbers to a common scale (the larger of the two)."""
        max_scale = max(a.scale, b.scale)
        a_aligned = ParsedScaledNumber(
            numerator=a.numerator * (10 ** (max_scale - a.scale)),
            scale=max_scale,
        )
        b_aligned = ParsedScaledNumber(
            numerator=b.numerator * (10 ** (max_scale - b.scale)),
            scale=max_scale,
        )
        return a_aligned, b_aligned

    @classmethod
    def add(cls, a: ParsedScaledNumber, b: ParsedScaledNumber) -> ParsedScaledNumber:
        """Add two scaled numbers with proper decimal alignment."""
        a_aligned, b_aligned = cls.align(a, b)
        return ParsedScaledNumber(
            numerator=a_aligned.numerator + b_aligned.numerator,
            scale=a_aligned.scale,
        )

    @classmethod
    def subtract(cls, a: ParsedScaledNumber, b: ParsedScaledNumber) -> ParsedScaledNumber:
        """Subtract two scaled numbers with proper decimal alignment."""
        a_aligned, b_aligned = cls.align(a, b)
        return ParsedScaledNumber(
            numerator=a_aligned.numerator - b_aligned.numerator,
            scale=a_aligned.scale,
        )

    @classmethod
    def multiply(cls, a: ParsedScaledNumber, b: ParsedScaledNumber) -> ParsedScaledNumber:
        """Multiply two scaled numbers. Scales add."""
        return ParsedScaledNumber(
            numerator=a.numerator * b.numerator,
            scale=a.scale + b.scale,
        )

    @staticmethod
    def from_float(value: float, max_scale: int = 9) -> ParsedScaledNumber:
        """Convert a float to a scaled number (for application-layer convenience).
        Example: 0.1 → ParsedScaledNumber(1, 1)
        """
        scale = 0
        while value != int(value) and scale < max_scale:
            value *= 10
            scale += 1
        return ParsedScaledNumber(numerator=int(value), scale=scale)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. THE BINARY GRID — Append‑only storage with geometric queries
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GridRecord:
    """A record stored in the grid with its bit address."""
    tokens: List[Token]
    bit_offset: int       # Absolute bit offset in the grid
    bit_length: int       # Length in bits (tokens * 5)
    parsed_values: List[ParsedNumber] = field(default_factory=list)

    @property
    def digit_vector(self) -> List[int]:
        """Return the concatenated signed digits of all numbers in this record."""
        vec = []
        for pv in self.parsed_values:
            vec.extend(pv.digits)
        return vec

    @property
    def value_vector(self) -> List[int]:
        """Return the integer values of all numbers in this record."""
        return [pv.value for pv in self.parsed_values]


class BinaryGrid:
    """A flat, contiguous, append-only sequence of 5-bit tokens.

    Storage: bit-addressable, O(1) seek and read.
    No mandatory file header.
    """

    def __init__(self):
        self._tokens: List[Token] = []
        self._records: List[GridRecord] = []
        self._bit_length: int = 0

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    @property
    def bit_length(self) -> int:
        return self._bit_length

    @property
    def record_count(self) -> int:
        return len(self._records)

    def append_tokens(self, tokens: List[Token]) -> int:
        """Append raw tokens to the grid. Returns the starting bit offset."""
        offset = self._bit_length
        self._tokens.extend(tokens)
        self._bit_length = len(self._tokens) * 5
        return offset

    def append_record(self, tokens: List[Token]) -> GridRecord:
        """Append a record (must end with RECORD token). Returns the GridRecord."""
        if not tokens or tokens[-1] != Token.RECORD:
            raise ValueError("Record must end with RECORD token")

        offset = self._bit_length
        self._tokens.extend(tokens)
        self._bit_length = len(self._tokens) * 5

        # Parse the record to extract values — use parser.output, not feed_tokens return
        parser = Parser()
        parser.feed_tokens(tokens)
        parser.finalize()
        numbers = [p for p in parser.output if isinstance(p, ParsedNumber)]

        record = GridRecord(
            tokens=list(tokens),
            bit_offset=offset,
            bit_length=len(tokens) * 5,
            parsed_values=numbers,
        )
        self._records.append(record)
        return record

    def get_record(self, index: int) -> GridRecord:
        """Get a record by index."""
        return self._records[index]

    def read_at(self, bit_offset: int, num_tokens: int) -> List[Token]:
        """Read num_tokens starting at the given bit offset. O(1)."""
        token_offset = bit_offset // 5
        if bit_offset % 5 != 0:
            raise ValueError("Bit offset must be aligned to 5-bit boundary")
        return self._tokens[token_offset:token_offset + num_tokens]

    def pack(self) -> Tuple[bytes, int]:
        """Serialize the entire grid to bytes."""
        return pack_to_bytes(self._tokens)

    @classmethod
    def from_packed(cls, data: bytes, pad_length: int) -> 'BinaryGrid':
        """Deserialize from packed bytes."""
        tokens = unpack_from_bytes(data, pad_length)
        grid = cls()
        grid._tokens = tokens
        grid._bit_length = len(tokens) * 5

        # Rebuild records by scanning for RECORD tokens
        current_record_start = 0
        record_parser = Parser()
        for i, t in enumerate(tokens):
            record_parser.feed(t)
            if t == Token.RECORD:
                record_tokens = tokens[current_record_start:i + 1]
                # Extract numbers accumulated since last RECORD
                numbers = [p for p in record_parser.output if isinstance(p, ParsedNumber)]
                record = GridRecord(
                    tokens=list(record_tokens),
                    bit_offset=current_record_start * 5,
                    bit_length=len(record_tokens) * 5,
                    parsed_values=list(numbers),
                )
                grid._records.append(record)
                # Reset for next record
                record_parser = Parser()
                current_record_start = i + 1

        return grid


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GEOMETRIC QUERIES — Hamming & Manhattan distance
# ═══════════════════════════════════════════════════════════════════════════════

def hamming_distance(addr1: int, addr2: int) -> int:
    """Number of bit positions where two addresses differ.

    Used for shard routing: find the shard whose starting address
    is closest to the target address.
    """
    return (addr1 ^ addr2).bit_count()


def manhattan_distance(vec1: List[int], vec2: List[int]) -> int:
    """Sum of absolute differences between corresponding elements of two vectors.

    For records of unequal length, pads the shorter with zeros.
    """
    n = max(len(vec1), len(vec2))
    v1 = list(vec1) + [0] * (n - len(vec1))
    v2 = list(vec2) + [0] * (n - len(vec2))
    return sum(abs(a - b) for a, b in zip(v1, v2))


def query_by_manhattan(grid: BinaryGrid, target: List[int], max_distance: int) -> List[GridRecord]:
    """Find all records whose digit-vector Manhattan distance from target < max_distance.

    This is the geometric equivalent of:
        SELECT * WHERE manhattan(value_vector, target) < max_distance
    """
    results = []
    for record in grid._records:
        vec = record.value_vector
        dist = manhattan_distance(vec, target)
        if dist < max_distance:
            results.append(record)
    return results


def query_by_hamming_shard(target_address: int, shard_addresses: List[int]) -> int:
    """Find the shard index whose address has minimum Hamming distance to target.

    Returns the index of the best shard.
    """
    best_idx = 0
    best_dist = hamming_distance(target_address, shard_addresses[0])
    for i, addr in enumerate(shard_addresses[1:], 1):
        dist = hamming_distance(target_address, addr)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CORRUPTION DETECTION & RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def inject_bit_flip(tokens: List[Token], position: int, bit_index: int) -> List[Token]:
    """Simulate a single bit-flip at the given token position and bit index (0-4)."""
    result = list(tokens)
    old_val = int(result[position])
    new_val = old_val ^ (1 << (4 - bit_index))
    result[position] = Token(new_val)
    return result


def find_next_sync_point(tokens: List[Token], start: int = 0) -> Optional[int]:
    """Find the next RECORD or CHECKSUM token after start (for resynchronization)."""
    for i in range(start, len(tokens)):
        if tokens[i] in (Token.RECORD, Token.CHECKSUM):
            return i
    return None


def scan_for_corruption(tokens: List[Token]) -> List[ChecksumResult]:
    """Scan a token stream, verifying all CHECKSUM markers.

    Returns a list of ChecksumResult for each CHECKSUM encountered.
    """
    results = []
    segment_start = 0

    i = 0
    while i < len(tokens):
        if tokens[i] == Token.CHECKSUM:
            # The checksum covers tokens since segment_start up to (but not including) CHECKSUM
            segment = tokens[segment_start:i]
            if i + 1 < len(tokens):
                expected = int(tokens[i + 1])
                result = verify_checksum(segment, expected)
                results.append(result)
                i += 2  # Skip CHECKSUM and its payload
                segment_start = i
            else:
                # CHECKSUM at end with no payload — malformed
                results.append(ChecksumResult(expected=-1, computed=compute_checksum(segment), passed=False))
                i += 1
                segment_start = i
        else:
            i += 1

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 11. HIGH-LEVEL API — Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════

class BinaryGridDB:
    """High-level interface for the Binary Grid Database."""

    def __init__(self):
        self.grid = BinaryGrid()
        self.encoder = Encoder()
        self.parser = Parser()

    def insert_number(self, value: int) -> List[Token]:
        """Insert a standalone number."""
        tokens = self.encoder.encode_integer(value)
        self.grid.append_tokens(tokens)
        return tokens

    def insert_word(self, text: str) -> List[Token]:
        """Insert a word."""
        tokens = self.encoder.encode_word(text)
        self.grid.append_tokens(tokens)
        return tokens

    def insert_record(self, *values: Union[int, str, List[Token]]) -> GridRecord:
        """Insert a record (values separated, terminated by RECORD)."""
        tokens = self.encoder.encode_record(*values)
        return self.grid.append_record(tokens)

    def insert_expression(self, *items: Union[int, str, List[int]]) -> int:
        """Insert an arithmetic expression and return its result."""
        tokens = self.encoder.encode_expression(list(items))
        self.grid.append_tokens(tokens)

        # Parse and evaluate
        parser = Parser()
        parsed = parser.feed_tokens(tokens)
        parser.finalize()
        return ArithmeticEvaluator.evaluate_parsed(parser.output)

    def query_manhattan(self, target: List[int], max_distance: int) -> List[GridRecord]:
        """Geometric query: records within Manhattan distance of target."""
        return query_by_manhattan(self.grid, target, max_distance)

    def pack(self) -> Tuple[bytes, int]:
        """Serialize the database to bytes."""
        return self.grid.pack()

    @classmethod
    def unpack(cls, data: bytes, pad_length: int) -> 'BinaryGridDB':
        """Deserialize from bytes."""
        db = cls()
        db.grid = BinaryGrid.from_packed(data, pad_length)
        return db

    def stats(self) -> dict:
        """Return database statistics."""
        return {
            'token_count': self.grid.token_count,
            'bit_length': self.grid.bit_length,
            'record_count': self.grid.record_count,
            'byte_length': (self.grid.bit_length + 7) // 8,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MAIN — Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("═" * 60)
    print("  BINARY GRID DATABASE — Demo")
    print("═" * 60)

    db = BinaryGridDB()

    # Example 1: Encoding numbers
    print("\n── Example 1: Number Encoding ──")
    for val in [123, -123, 0, -5, 42]:
        tokens = Encoder.encode_integer(val)
        bs = token_stream_to_binary_string(tokens)
        print(f"  {val:>5} → {bs}")

    # Example 2: Encoding words
    print("\n── Example 2: Word Encoding ──")
    for word in ["HI", "HELLO", "CLAUDE"]:
        tokens = Encoder.encode_word(word)
        bs = token_stream_to_binary_string(tokens)
        print(f"  {word:>6} → {bs}")

    # Example 3: Record boundaries
    print("\n── Example 3: Records ──")
    r1 = db.insert_record(1, 2)
    r2 = db.insert_record(-123, 8175)
    print(f"  Record 1: {r1.value_vector} @ bit {r1.bit_offset}")
    print(f"  Record 2: {r2.value_vector} @ bit {r2.bit_offset}")

    # Example 4: Arithmetic
    print("\n── Example 4: Arithmetic ──")
    # Multi-digit numbers are passed as lists of signed digits
    result = db.insert_expression([-1, -2, -3], '*', [-8, -1, -7, -5])
    print(f"  -123 * -8175 = {result:,}")  # Expected: 1,005,525

    # Also test simpler expressions
    r2 = db.insert_expression([2], '+', [-3])
    print(f"  2 + (-3) = {r2}")

    # Scale (S) is a storage annotation, NOT an arithmetic operator.
    # It marks "this integer has N implied decimal places."
    # -1234 S 3 means: integer -1234, 3 decimal places → -1.234
    print(f"\n  Scale annotation demo:")
    scaled_tokens = Encoder.encode_expression([[-1, -2, -3, -4], 'S', [3]])
    parser = Parser()
    parser.feed_tokens(scaled_tokens)
    parser.finalize()
    resolved = resolve_scaled_numbers(parser.output)
    for r in resolved:
        print(f"    {r}")
    # The database stores integers; the application layer interprets scale

    # Example 4b: Decimal Arithmetic (application layer)
    print("\n── Example 4b: Decimal Arithmetic (Application Layer) ──")
    from binary_grid_db import DecimalArithmetic
    # Store 0.1 as (1, scale=1) and 0.02 as (2, scale=2)
    a = ParsedScaledNumber(1, 1)   # 0.1
    b = ParsedScaledNumber(2, 2)   # 0.02
    print(f"  {a} + {b}")
    result = DecimalArithmetic.add(a, b)
    print(f"  = {result}")
    # 0.1 + 0.02: align to scale 2 → (10, 2) + (2, 2) = (12, 2) = 0.12
    print(f"  Aligned: (1→10, scale=2) + (2, scale=2) = (12, scale=2)")
    print(f"  Interpreted: 12 / 10² = {result.as_float}")

    # Example 5: Serialization round-trip
    print("\n── Example 5: Serialization ──")
    raw_tokens = Encoder.encode_integer(123) + Encoder.encode_word("HI")
    packed, pad = pack_to_bytes(raw_tokens)
    unpacked = unpack_from_bytes(packed, pad)
    print(f"  Original: {token_stream_to_binary_string(raw_tokens)}")
    print(f"  Packed:   {packed.hex()} ({len(packed)} bytes, pad={pad})")
    print(f"  Roundtrip: {'✓' if raw_tokens == unpacked else '✗'}")

    # Example 6: Checksum
    print("\n── Example 6: Checksum ──")
    data_tokens = Encoder.encode_integer(123) + Encoder.encode_integer(456)
    cs_value = compute_checksum(data_tokens)
    print(f"  Checksum of [123, 456]: {cs_value} (mod 32)")
    cs_tokens = append_checksum(data_tokens)
    print(f"  With checksum: {token_stream_to_binary_string(cs_tokens)}")

    # Example 7: Manhattan distance query
    print("\n── Example 7: Geometric Query ──")
    db2 = BinaryGridDB()
    db2.insert_record(1, 2, 3)
    db2.insert_record(2, 3, 4)
    db2.insert_record(10, 20, 30)
    db2.insert_record(0, 1, 2)

    target = [1, 2, 3]
    results = db2.query_manhattan(target, 10)
    print(f"  Records within Manhattan distance < 10 of {target}:")
    for r in results:
        d = manhattan_distance(r.value_vector, target)
        print(f"    {r.value_vector} (distance={d})")

    # Example 8: Hamming distance
    print("\n── Example 8: Hamming Distance ──")
    shards = [0b00000, 0b11111, 0b10101]
    target_addr = 0b10100
    best = query_by_hamming_shard(target_addr, shards)
    print(f"  Target address: {target_addr:05b}")
    for i, s in enumerate(shards):
        hd = hamming_distance(target_addr, s)
        marker = " ← BEST" if i == best else ""
        print(f"    Shard {i}: {s:05b} (Hamming={hd}){marker}")

    # Example 9: Corruption detection
    print("\n── Example 9: Corruption Detection ──")
    clean_tokens = Encoder.encode_expression([[-1, -2, -3], '*', [-8, -1, -7, -5]])
    cs_tokens = append_checksum(clean_tokens)
    print(f"  Clean checksum: {scan_for_corruption(cs_tokens)}")

    corrupted = inject_bit_flip(cs_tokens, position=0, bit_index=2)
    print(f"  After bit-flip:  {scan_for_corruption(corrupted)}")

    # Example 10: Parser state machine
    print("\n── Example 10: Parser State Machine ──")
    parser = Parser()
    mixed = (
        Encoder.encode_integer(42) +
        Encoder.encode_word("HI") +
        Encoder.encode_integer(-7)
    )
    parsed = parser.feed_tokens(mixed)
    parser.finalize()
    for p in parser.output:
        print(f"  {p}")

    print("\n" + "═" * 60)
    print("  Demo complete. All systems operational.")
    print("═" * 60)
