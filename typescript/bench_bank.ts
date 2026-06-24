/**
 * MicroBank benchmark — TypeScript engine, with the BATCH (group-commit) path.
 *
 * Three ledgers, identical seeded workload:
 *   1. GridDB single   — AllocGrid, one fsync PER write (what the Python bench did)
 *   2. GridDB batch    — GroupCommitAllocGrid, one fsync PER BATCH  <-- the property
 *   3. SQLite batch    — node:sqlite, BATCH transfers per BEGIN/COMMIT (one fsync/batch)
 *
 * Money is integer cents. Each transfer moves cents between two accounts.
 * We verify money conservation (read back from disk) and report throughput + disk.
 *
 * Run: node_modules/.bin/ts-node -T bench_bank.ts [users] [transfers] [batch]
 */
import fs from 'fs';
import path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { AllocGrid, GroupCommitAllocGrid } from './src/alloc';
import { Encoder } from './src/encoder';
import { Token, ParsedToken } from './src/types';

const N = parseInt(process.argv[2] || '2000', 10);
const M = parseInt(process.argv[3] || '10000', 10);
const BATCH = parseInt(process.argv[4] || '200', 10);
const START = 100_00; // $100.00 in cents
const ROOT = path.join(__dirname, '_bench_bank');

// deterministic RNG (mulberry32) so all three ledgers see identical transfers
function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function uname(i: number): string {
  return 'USER' + String(i).split('').map(d => String.fromCharCode(65 + +d)).join('');
}
function encAccount(name: string, cents: number): Token[] {
  return Encoder.encodeRecord(name, cents);
}
function decode(parsed: ParsedToken[]): [string, number] {
  let w: string | null = null, n: number | null = null;
  for (const p of parsed as any[]) {
    if (p.type === 'word' && w === null) w = p.text;
    else if (p.type === 'number' && n === null) n = p.value;
  }
  return [w as string, n as number];
}
function dirOf(tag: string) {
  const d = path.join(ROOT, tag);
  fs.rmSync(d, { recursive: true, force: true });
  fs.mkdirSync(d, { recursive: true });
  return d;
}
function diskBytes(dir: string): number {
  let total = 0;
  for (const f of fs.readdirSync(dir)) {
    const p = path.join(dir, f);
    const st = fs.statSync(p);
    if (st.isFile()) total += st.size;
  }
  return total;
}
const now = () => Number(process.hrtime.bigint()) / 1e6; // ms

interface Res {
  name: string; createOps: number; xferOps: number;
  applied: number; disk: number; total: number; expected: number;
}

// ── Grid ledger (works for both single + group-commit via the write fn) ──────
function runGrid(name: string, dir: string, grid: any, flush: (() => void) | null,
                 n: number = N, m: number = M): Res {
  const rand = mulberry32(42);
  // 1) create accounts
  let t0 = now();
  for (let i = 0; i < n; i++) grid.write(i, encAccount(uname(i), START));
  if (flush) flush();
  const createOps = n / ((now() - t0) / 1000);

  // dirty overlay so reads within an unflushed batch stay correct
  const dirty = new Map<number, [string, number]>();
  const readAcct = (rid: number): [string, number] => {
    if (dirty.has(rid)) return dirty.get(rid)!;
    return decode(grid.read(rid).parsed);
  };

  // 2) transfers
  t0 = now();
  let applied = 0;
  for (let k = 0; k < m; k++) {
    const a = Math.floor(rand() * n), b = Math.floor(rand() * n);
    if (a === b) continue;
    const amt = 1 + Math.floor(rand() * 500);
    const [na, ba] = readAcct(a);
    const [nb, bb] = readAcct(b);
    if (ba < amt) continue;
    dirty.set(a, [na, ba - amt]);
    dirty.set(b, [nb, bb + amt]);
    grid.write(a, encAccount(na, ba - amt));
    grid.write(b, encAccount(nb, bb + amt));
    applied++;
    if (flush && (k + 1) % BATCH === 0) { flush(); dirty.clear(); }
  }
  if (flush) flush();
  dirty.clear();
  const xferOps = applied / ((now() - t0) / 1000);

  // 3) conservation (read back from disk)
  let total = 0;
  for (let i = 0; i < n; i++) total += decode(grid.read(i).parsed)[1];

  return { name, createOps, xferOps, applied, disk: diskBytes(dir), total, expected: n * START };
}

// ── SQLite ledger, batched transactions ─────────────────────────────────────
function runSqlite(dir: string): Res {
  const db = new DatabaseSync(path.join(dir, 'bank.db'));
  db.exec('PRAGMA journal_mode=WAL');
  db.exec('PRAGMA synchronous=FULL');
  db.exec('CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER NOT NULL)');
  const ins = db.prepare('INSERT INTO accounts(id,balance) VALUES(?,?)');
  const sel = db.prepare('SELECT balance FROM accounts WHERE id=?');
  const upd = db.prepare('UPDATE accounts SET balance=? WHERE id=?');
  const rand = mulberry32(42);

  let t0 = now();
  db.exec('BEGIN');
  for (let i = 0; i < N; i++) ins.run(i, START);
  db.exec('COMMIT');
  const createOps = N / ((now() - t0) / 1000);

  t0 = now();
  let applied = 0, inBatch = 0;
  db.exec('BEGIN');
  for (let k = 0; k < M; k++) {
    const a = Math.floor(rand() * N), b = Math.floor(rand() * N);
    if (a === b) continue;
    const amt = 1 + Math.floor(rand() * 500);
    const ba = (sel.get(a) as any).balance as number;
    const bb = (sel.get(b) as any).balance as number;
    if (ba < amt) continue;
    upd.run(ba - amt, a);
    upd.run(bb + amt, b);
    applied++;
    if (++inBatch >= BATCH) { db.exec('COMMIT'); db.exec('BEGIN'); inBatch = 0; }
  }
  db.exec('COMMIT');
  const xferOps = applied / ((now() - t0) / 1000);

  const total = (db.prepare('SELECT COALESCE(SUM(balance),0) s FROM accounts').get() as any).s as number;
  db.close();
  return { name: 'SQLite (batch)', createOps, xferOps, applied, disk: diskBytes(dir), total, expected: N * START };
}

function fmtBytes(n: number) {
  const u = ['B', 'KB', 'MB', 'GB']; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(1)} ${u[i]}`;
}

function main() {
  console.log('='.repeat(72));
  console.log(`  MicroBank (TypeScript) — ${N} users, ${M} transfers, batch=${BATCH}`);
  console.log('='.repeat(72));

  const results: Res[] = [];

  // 1) grid, one fsync per write — capped small (3 fsync/write is fsync-bound on macOS)
  const SN = Math.min(N, 300), SM = Math.min(M, 800);
  {
    const d = dirOf('grid_single');
    const g = new AllocGrid(d);
    results.push(runGrid(`GridDB per-write*`, d, g, null, SN, SM));
  }
  // 2) grid, group commit (the batch property)
  {
    const d = dirOf('grid_batch');
    const g = new GroupCommitAllocGrid(d, 10_000_000); // never auto-flush; we flush manually
    results.push(runGrid('GridDB (batch)', d, g, () => g.flush()));
  }
  // 3) sqlite batched
  results.push(runSqlite(dirOf('sqlite')));

  const pad = (s: string, w: number) => s.padStart(w);
  console.log();
  console.log('Backend'.padEnd(22) + pad('Create/s', 12) + pad('Transfer/s', 12) + pad('Disk', 11) + pad('Conserved', 11));
  console.log('-'.repeat(68));
  for (const r of results) {
    const ok = r.total === r.expected ? 'YES' : 'NO!';
    console.log(
      r.name.padEnd(22) +
      pad(Math.round(r.createOps).toLocaleString(), 12) +
      pad(Math.round(r.xferOps).toLocaleString(), 12) +
      pad(fmtBytes(r.disk), 11) +
      pad(ok, 11)
    );
  }

  console.log();
  console.log(`  Conservation (sum==expected) — ` +
    results.map(r => `${r.name.split(' ')[0]}:${r.total.toLocaleString()}/${r.expected.toLocaleString()}`).join('  '));
  console.log(`  * per-write capped at ${SN} users / ${SM} transfers (it is fsync-bound); throughput is per-second so still comparable.`);

  const [single, batch, sql] = results;
  console.log();
  console.log('='.repeat(72));
  console.log('  VERDICT');
  console.log('='.repeat(72));
  console.log(`  Batch vs per-write (grid): ${(batch.xferOps / single.xferOps).toFixed(1)}x faster transfers ` +
    `(${Math.round(batch.xferOps).toLocaleString()} vs ${Math.round(single.xferOps).toLocaleString()} tx/s)`);
  const xWin = batch.xferOps >= sql.xferOps ? 'GridDB(batch)' : 'SQLite';
  const xr = Math.max(batch.xferOps, sql.xferOps) / Math.min(batch.xferOps, sql.xferOps);
  console.log(`  Grid(batch) vs SQLite:     ${xWin} faster by ${xr.toFixed(1)}x ` +
    `(${Math.round(batch.xferOps).toLocaleString()} vs ${Math.round(sql.xferOps).toLocaleString()} tx/s)`);
  const dWin = batch.disk <= sql.disk ? 'GridDB' : 'SQLite';
  console.log(`  Disk:                      ${dWin} smaller ` +
    `(${fmtBytes(batch.disk)} grid vs ${fmtBytes(sql.disk)} sqlite)`);
  const allOk = results.every(r => r.total === r.expected);
  console.log(`  Correctness:               ${allOk ? 'all conserve money' : 'MISMATCH'}`);
  console.log('='.repeat(72));
}

main();
