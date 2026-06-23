#!/bin/bash
# GridDB Cross-Language Conformance Test
# ======================================
# Runs Python and TypeScript batteries, compares byte-identical output.
# Exits 0 if all match, 1 if any diverge.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0

echo "══════════════════════════════════════════════"
echo "  GridDB Cross-Language Conformance Test"
echo "══════════════════════════════════════════════"

# Run Python
echo ""
echo "── Python ──"
PY_OUT=$(mktemp)
python3 "$SCRIPT_DIR/python/battery_py.py" > "$PY_OUT"
echo "  Python: $(wc -l < "$PY_OUT") lines"

# Run TypeScript
echo ""
echo "── TypeScript ──"
TS_OUT=$(mktemp)
cd "$SCRIPT_DIR/typescript"
npx tsx tests/battery_ts.ts > "$TS_OUT" 2>/dev/null
echo "  TypeScript: $(wc -l < "$TS_OUT") lines"

# Compare
echo ""
echo "── Comparison ──"

python3 -c "
import json, sys

py = json.load(open('$PY_OUT'))
ts = json.load(open('$TS_OUT'))

categories = sorted(set(list(py.keys()) + list(ts.keys())))
pass_count = 0
fail_count = 0

for cat in categories:
    pv = py.get(cat, [])
    tv = ts.get(cat, [])
    for i in range(max(len(pv), len(tv))):
        pi = pv[i] if i < len(pv) else None
        ti = tv[i] if i < len(tv) else None
        label = f'{cat}[{i}]'
        inp = (pi or ti).get('input', '?')

        if pi and pi.get('error'):
            print(f'  ✗ {label}: {inp} — PY ERROR: {pi[\"error\"]}')
            fail_count += 1
        elif ti and ti.get('error'):
            print(f'  ✗ {label}: {inp} — TS ERROR: {ti[\"error\"]}')
            fail_count += 1
        elif not pi or not ti:
            print(f'  ✗ {label}: {inp} — MISSING')
            fail_count += 1
        elif pi.get('hex') != ti.get('hex'):
            print(f'  ✗ {label}: {inp} — HEX MISMATCH')
            print(f'     PY: {pi[\"hex\"][:40]}')
            print(f'     TS: {ti[\"hex\"][:40]}')
            fail_count += 1
        else:
            pass_count += 1

print(f'\n  Results: {pass_count} pass, {fail_count} fail')
if fail_count > 0:
    print(f'  ✗ CROSS-LANGUAGE DIVERGENCE DETECTED')
    sys.exit(1)
else:
    print(f'  ✓ ALL BYTE-IDENTICAL — Cross-language determinism verified')
"

STATUS=$?
rm -f "$PY_OUT" "$TS_OUT"

echo ""
echo "══════════════════════════════════════════════"
if [ $STATUS -eq 0 ]; then
    echo "  PASS — Python ≡ TypeScript, bit for bit"
else
    echo "  FAIL — divergence found"
fi
echo "══════════════════════════════════════════════"
exit $STATUS
