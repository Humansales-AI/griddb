#!/usr/bin/env python3
"""
GridDB Self-Distributing Archive
=================================
One file: griddb-v4.grid
Contains the entire GridDB library as GridDB records.

Pack:   python3 griddb_pack.py pack   → griddb-v4.grid
Unpack: python3 griddb_pack.py unpack → writes all files

Uses GridDB's own alloc+data format to store itself.
"""
import os, sys, struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid

COMPONENTS = [
    'griddb_pack.py',          # ← the unpacker itself
    'binary_grid_db.py',
    'griddb_alloc.py',
    'griddb_wal.py',
    'griddb_positioned.py',
    'griddb_index.py',
    'griddb_transactions.py',
    'griddb_replication.py',
    'griddb_changestream.py',
    'griddb_correctness.py',
    'test_binary_grid_db.py',
]

OUTPUT_FILE = 'griddb-v4.grid'

def pack(directory: str, output: str = OUTPUT_FILE):
    """Pack all component files into a single GridDB archive."""
    grid = AllocGrid(data_dir=os.path.join(directory, '_pack_tmp'))
    rid = 0

    print(f"Packing {len(COMPONENTS)} files → {output}")
    for filename in COMPONENTS:
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP: {filename} (not found)")
            continue

        with open(filepath, 'rb') as f:
            content = f.read()

        # Record: WORD(filename) NUM(file_size) RECORD
        safe_name = filename.replace('_', '-').lower()
        name_tokens = [*Encoder.encode_word(safe_name), *Encoder.encode_integer(len(content)), Token.RECORD]
        grid.write(rid, name_tokens)
        rid += 1

        # Store content: each byte as a NUM token
        # Packed as: NUM(b0) NUM(b1) ... NUM(bN) RECORD
        content_tokens = []
        for b in content:
            content_tokens.extend(Encoder.encode_integer(b))
        content_tokens.append(Token.RECORD)
        grid.write(rid, content_tokens)
        rid += 1

        print(f"  {filename}: {len(content)} bytes, {len(content_tokens)} tokens")

    # Copy alloc+data files to output location
    tmp_dir = os.path.join(directory, '_pack_tmp')
    alloc_src = os.path.join(tmp_dir, 'alloc.grid')
    data_src = os.path.join(tmp_dir, 'data.grid')

    # Merge into one file: [header] [alloc] [data]
    with open(output, 'wb') as out:
        # Header: magic + version + file_count + alloc_size
        with open(alloc_src, 'rb') as af:
            alloc_data = af.read()
        with open(data_src, 'rb') as df:
            data_data = df.read()
        out.write(struct.pack('>4sIII', b'GRDB', 4, rid, len(alloc_data)))
        out.write(alloc_data)
        out.write(data_data)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    size = os.path.getsize(output)
    print(f"  Done: {output} ({size:,} bytes)")


def unpack(archive: str, output_dir: str):
    """Unpack a GridDB archive into component files."""
    if not os.path.exists(archive):
        print(f"Archive not found: {archive}")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(archive, 'rb') as f:
        magic, version, file_count, alloc_size = struct.unpack('>4sIII', f.read(16))
        if magic != b'GRDB':
            print(f"Invalid archive: {magic}")
            return

        print(f"GridDB archive v{version}, {file_count} records, alloc={alloc_size}B")

        remaining = f.read()

    alloc_dir = os.path.join(output_dir, '_unpack_tmp')
    os.makedirs(alloc_dir, exist_ok=True)
    alloc_path = os.path.join(alloc_dir, 'alloc.grid')
    data_path = os.path.join(alloc_dir, 'data.grid')

    alloc_bytes = remaining[:alloc_size]
    data_bytes = remaining[alloc_size:]

    with open(alloc_path, 'wb') as f:
        f.write(alloc_bytes)
    with open(data_path, 'wb') as f:
        f.write(data_bytes)

    # Read via AllocGrid
    grid = AllocGrid(data_dir=alloc_dir)
    extracted = 0
    for rid in range(0, file_count, 2):
        name_rec = grid.read(rid)
        content_rec = grid.read(rid + 1)
        if not name_rec or not content_rec:
            continue

        # Parse filename
        words = [p.text for p in name_rec.parsed if isinstance(p, ParsedWord)]
        nums  = [p.value for p in name_rec.parsed if isinstance(p, ParsedNumber)]
        filename = ''.join(words).replace('-', '_')
        file_size = nums[-1] if nums else 0

        # Parse content: each NUM token is a byte
        content_nums = [p.value for p in content_rec.parsed if isinstance(p, ParsedNumber)]
        content_bytes = bytes(content_nums)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(content_bytes)
        print(f"  {filename}: {len(content_bytes)} bytes")
        extracted += 1

    import shutil
    shutil.rmtree(alloc_dir, ignore_errors=True)
    print(f"  Done: {extracted} files → {output_dir}/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 griddb_pack.py pack|unpack [directory]")
        sys.exit(1)

    cmd = sys.argv[1]
    cwd = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))

    if cmd == 'pack':
        pack(cwd)
    elif cmd == 'unpack':
        unpack(os.path.join(cwd, OUTPUT_FILE) if os.path.isdir(cwd) else cwd,
               sys.argv[3] if len(sys.argv) > 3 else './unpacked')
    else:
        print(f"Unknown command: {cmd}")
