#!/bin/bash
# GridDB — one-command verification of the load-bearing claims.
# ===========================================================
#   1. Cross-language determinism — Python ≡ TypeScript, bit for bit
#   2. Canonical compaction       — compact() reclaims space AND produces
#                                   byte-identical output to a fresh build
#                                   (logical content -> same bytes, any history)
#   3. Lock-free concurrent reads — readers see consistent snapshots while a
#                                   writer runs; zero torn reads
#
# Exits 0 only if every check passes.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATUS=0

echo "════════════════════════════════════════════════════════"
echo "  GridDB — Verification Suite"
echo "════════════════════════════════════════════════════════"

# ── 1. Cross-language conformance ───────────────────────────
echo ""
echo "── [1/3] Cross-language determinism ──"
if bash "$SCRIPT_DIR/conformance.sh" >/tmp/griddb_conf.$$ 2>&1; then
    grep -E "Results:|byte-identical|BYTE-IDENTICAL" /tmp/griddb_conf.$$ | sed 's/^/  /'
    echo "  ✓ conformance PASS"
else
    cat /tmp/griddb_conf.$$
    echo "  ✗ conformance FAIL"
    STATUS=1
fi
rm -f /tmp/griddb_conf.$$

# ── 2. Canonical compaction (both engines) ──────────────────
echo ""
echo "── [2/3] Canonical compaction ──"

echo "  Python:"
if python3 "$SCRIPT_DIR/python/verify_compact.py"; then
    echo "  ✓ python compaction PASS"
else
    echo "  ✗ python compaction FAIL"
    STATUS=1
fi

echo "  TypeScript:"
if ( cd "$SCRIPT_DIR/typescript" && npx tsx tests/verify_compact.ts ); then
    echo "  ✓ typescript compaction PASS"
else
    echo "  ✗ typescript compaction FAIL"
    STATUS=1
fi

# ── 3. Lock-free concurrent reads ───────────────────────────
echo ""
echo "── [3/3] Lock-free concurrent reads ──"
if python3 "$SCRIPT_DIR/python/verify_concurrency.py"; then
    echo "  ✓ concurrency PASS"
else
    echo "  ✗ concurrency FAIL"
    STATUS=1
fi

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
if [ $STATUS -eq 0 ]; then
    echo "  PASS — determinism + canonical compaction + lock-free reads verified"
else
    echo "  FAIL — see output above"
fi
echo "════════════════════════════════════════════════════════"
exit $STATUS
