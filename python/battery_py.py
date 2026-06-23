#!/usr/bin/env python3
"""
GridDB Cross-Language Conformance Battery — Python
====================================================
Encodes 40+ test vectors and outputs JSON with packed hex.
Compared against battery_ts.ts for byte-identical determinism.

Run: python3 battery_py.py > py_output.json
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_grid_db import Encoder, pack_to_bytes

TESTS = {
    "integers": [0, 1, -1, 7, -5, 42, 123, -123, 999, -9999,
                 2147483647, -2147483648, 1000000, 999999999],
    "bigIntegers": [12345678901234567890, 10**30],
    "bigints": [10**30, 999999999999999999999999999999],
    "words": ["A", "Z", "HELLO", "REC", "hello", "aB", "Ba", "abcXYZ",
              "HelloWorld", "test@example.com", "a.b-c", "x@y", "a-b"],
    "special": ["a@b", "a-b", "x.y", "a.b-c", "x@y.z"],
    "special2": ["a!b", "c#d", "(x)", "a+b", "key=val", "a/b", "[z]", "p;q",
                 "a$b", "x%y", "a&b", "x*y", "a_b", "x|y", "a^b", "{x}", "`y`"],
}

results = {}
for category, values in TESTS.items():
    results[category] = []
    for v in values:
        try:
            if isinstance(v, int):
                tokens = Encoder.encode_integer(v)
            else:
                tokens = Encoder.encode_word(v)
            packed, pad = pack_to_bytes(tokens)
            results[category].append({
                "input": str(v),
                "hex": bytes(packed).hex(),
                "tokens": len(tokens),
                "pad": pad,
            })
        except Exception as e:
            results[category].append({
                "input": str(v),
                "error": str(e),
            })

print(json.dumps(results, indent=2))
