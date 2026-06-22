#!/usr/bin/env python3
"""
GridDB Self-Contained Unpacker — zero dependencies, standalone
===============================================================
Unpacks griddb-v4.grid. No GridDB installation needed. Just Python 3.

Usage: python3 griddb_pack.py unpack griddb-v4.grid ./output/
"""
import os, sys, struct

# ═══════════════════════════════════════════════════════════════════════
# Minimal inline 5-bit decoder — no imports needed
# ═══════════════════════════════════════════════════════════════════════

T_END, T_RECORD, T_START = 30, 28, 31
WORD_CH = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
           'Q','R','S','T','U','V','W','X','Y','Z',' ','.']
SP_CH   = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
           'q','r','s','t','u','v','w','x','y','z','@','-']

def _unpack(data, pad, n):
    bits = []; [bits.extend([(b>>i)&1 for i in range(7,-1,-1)]) for b in data]
    bits = bits[:len(bits)-pad]
    return [sum(bits[i+j]<<(4-j) for j in range(5)) for i in range(0,len(bits)-4,5)][:n]

def _parse_val(tokens, i):
    digs, j = [], i
    while j < len(tokens):
        t = tokens[j]
        if 0 <= t <= 9: digs.append(t)
        elif 17 <= t <= 25: digs.append(-(t-16))
        else: break
        j += 1
    if not digs: return 0, i
    v, n = 0, len(digs)
    for k, d in enumerate(digs): v += d * (10**(n-1-k))
    return v, j + (1 if j < len(tokens) and tokens[j] == T_END else 0)

def _parse_words(tokens):
    words, cur, st = [], [], 'N'
    for t in tokens:
        if t == T_START:
            if st == 'N': st = 'W'
            elif st == 'W': words.append(''.join(cur)); cur = []; st = 'S'
        elif t == T_END:
            if cur: words.append(''.join(cur)); cur = []
            st = 'W' if st == 'S' else 'N'
        elif t == T_RECORD:
            if cur: words.append(''.join(cur)); break
        elif st == 'W' and 0 <= t <= 27: cur.append(WORD_CH[t])
        elif st == 'S' and 0 <= t <= 27: cur.append(SP_CH[t])
        elif 0 <= t <= 9 or 17 <= t <= 25: pass  # digit in word = metadata
    if cur: words.append(''.join(cur))
    return words

def _all_nums(tokens):
    nums, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if 0 <= t <= 9 or 17 <= t <= 25:
            v, ni = _parse_val(tokens, i); nums.append(v); i = ni
        else: i += 1
    return nums

# ═══════════════════════════════════════════════════════════════════════

def unpack(archive, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(archive, 'rb') as f:
        magic, ver, fc, asz = struct.unpack('>4sIII', f.read(16))
        if magic != b'GRDB': print("Invalid archive"); return
        rest = f.read()
    alloc, data = rest[:asz], rest[asz:]
    DS = 12  # data file header size
    print(f"GridDB v{ver}, {fc} records")

    for rid in range(0, fc, 2):
        # Name record
        ae = 8 + rid * 16
        if ae + 16 > len(alloc): break
        off = int.from_bytes(alloc[ae:ae+8], 'big')
        bl = struct.unpack('>I', alloc[ae+8:ae+12])[0]
        fl = struct.unpack('>I', alloc[ae+12:ae+16])[0]
        if fl == 0: continue
        raw = data[off-DS : off-DS + (bl+7)//8]
        exp = bl // 5
        tok = None
        for p in range(8):
            try:
                t = _unpack(raw, p, exp)
                if len(t) == exp: tok = t; break
            except: pass
        if not tok: continue
        words = _parse_words(tok)
        nums = _all_nums(tok)
        fname = ''.join(words).replace('-', '_')
        fsize = nums[-1] if nums else 0

        # Content record
        ae2 = 8 + (rid+1) * 16
        if ae2 + 16 > len(alloc): break
        off2 = int.from_bytes(alloc[ae2:ae2+8], 'big')
        bl2 = struct.unpack('>I', alloc[ae2+8:ae2+12])[0]
        fl2 = struct.unpack('>I', alloc[ae2+12:ae2+16])[0]
        if fl2 == 0: continue
        raw2 = data[off2-DS : off2-DS + (bl2+7)//8]
        exp2 = bl2 // 5
        tok2 = None
        for p in range(8):
            try:
                t = _unpack(raw2, p, exp2)
                if len(t) == exp2: tok2 = t; break
            except: pass
        if not tok2: continue
        content = bytes(b for b in _all_nums(tok2) if 0 <= b <= 255)

        with open(os.path.join(out_dir, fname), 'wb') as f:
            f.write(content)
        print(f"  {fname}: {len(content)} bytes")
    print(f"  Done → {out_dir}/")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 griddb_pack.py unpack <archive> [outdir]")
        sys.exit(1)
    unpack(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else './griddb')
