#!/usr/bin/env python3
"""
Fun test: learned geometric retrieval on the 5bit grid vs exact SQL on SQLite.

Pipeline
  1. Build a fixed dataset of N 3-int records, deterministically.
  2. Store it in a real 5bit AllocGrid (factory token format) AND in SQLite.
  3. Train the numpy GridTransformer — on records read BACK from the grid —
     to answer "which records are within Manhattan distance T of a query?"
  4. Held-out queries: ask both systems the same question.
       - 5bit  : transformer reads grid token streams, predicts matches
       - SQLite: exact  SELECT ... WHERE |a-qa|+|b-qb|+|c-qc| < T   (ground truth)
  5. Compare accuracy + speed. Verify determinism (same seed -> same bytes/logits).

Honest by construction: SQLite's answer IS the ground truth (exact). The
question is how close a 32-token transformer running on raw grid bytes gets,
and what each costs.
"""
import os
import sys
import time
import sqlite3
import tempfile
import shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

from binary_grid_db import Encoder, Token                       # noqa: E402
from griddb_alloc import AllocGrid                              # noqa: E402
from grid_transformer import (                                  # noqa: E402
    GridTransformer, binary_cross_entropy, _sgd_step,
    _accumulate_embed_grads, generate_query_tokens,
)

SEED = 42
N = 64           # dataset records
VRANGE = 10      # values 0..9
THRESH = 6       # Manhattan match threshold
K = 10           # candidates presented per query
SEQ = 96
STEPS = 1500
BATCH = 24


def make_query(rng, data):
    """Query anchored near a random dataset point so neighbors exist."""
    base = data[rng.randint(len(data))]
    return np.clip(base + rng.randint(-3, 4, size=3), 0, VRANGE - 1)


def balanced_candidates(rng, data, q):
    """K candidate rids, ~half within THRESH and half outside, so training
    isn't degenerate (matches what the original generate_batch did)."""
    d = np.abs(data - q).sum(1)
    match = np.where(d < THRESH)[0]
    non = np.where(d >= THRESH)[0]
    n_pos = min(len(match), K // 2)
    n_neg = min(len(non), K - n_pos)
    pos = rng.choice(match, size=n_pos, replace=False) if n_pos else np.array([], int)
    neg = rng.choice(non, size=n_neg, replace=False) if n_neg else np.array([], int)
    rids = np.concatenate([pos, neg]).astype(int)
    rng.shuffle(rids)
    return [int(r) for r in rids]


def decode_vec(rec):
    """Pull the 3-int vector out of a grid record (exclude control Tokens)."""
    return [p.value for p in rec.parsed
            if getattr(p, "value", None) is not None and not isinstance(p, Token)]


def record_token_ids(rec):
    return [int(t) for t in rec.tokens]


def build_example(rng, grid, rids, q):
    """[query tokens][K record token streams read FROM the grid], + per-pos labels."""
    tok = list(generate_query_tokens(int(q[0]), int(q[1]), int(q[2])))
    spans = []  # (start, end, is_match)
    for rid in rids:
        rec = grid.read(rid)
        v = decode_vec(rec)
        start = len(tok)
        tok.extend(record_token_ids(rec))
        end = len(tok)
        dist = sum(abs(a - b) for a, b in zip(v, q))
        spans.append((start, end, 1.0 if dist < THRESH else 0.0))
    ids = np.zeros(SEQ, dtype=np.int32)
    lab = np.zeros(SEQ, dtype=np.float32)
    msk = np.zeros(SEQ, dtype=np.float32)
    tok = tok[:SEQ]
    for i, t in enumerate(tok):
        ids[i] = t; msk[i] = 1.0
    for s, e, m in spans:
        for i in range(s, min(e, SEQ)):
            lab[i] = m
    return ids, lab, msk, spans


def main():
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    root = tempfile.mkdtemp(prefix="grid_vs_sqlite_")
    try:
        # ── 1+2. dataset → 5bit grid AND sqlite ──────────────────────────
        data = rng.randint(0, VRANGE, size=(N, 3))
        grid = AllocGrid(data_dir=os.path.join(root, "grid"), cache_size=N)
        for i, (a, b, c) in enumerate(data):
            grid.write(i, Encoder.encode_record(int(a), int(b), int(c)))

        db = sqlite3.connect(os.path.join(root, "d.db"))
        db.execute("CREATE TABLE recs(id INT PRIMARY KEY, a INT, b INT, c INT)")
        db.executemany("INSERT INTO recs VALUES(?,?,?,?)",
                       [(i, int(a), int(b), int(c)) for i, (a, b, c) in enumerate(data)])
        db.commit()
        print(f"  stored {N} records in 5bit grid + SQLite (deterministic, seed={SEED})")

        # ── 3. train transformer on records read FROM the grid ───────────
        model = GridTransformer(d_model=64, n_heads=4, d_ff=256, n_blocks=2,
                                max_seq_len=SEQ)
        t0 = time.perf_counter()
        for step in range(STEPS):
            ids = np.zeros((BATCH, SEQ), dtype=np.int32)
            lab = np.zeros((BATCH, SEQ), dtype=np.float32)
            msk = np.zeros((BATCH, SEQ), dtype=np.float32)
            for b in range(BATCH):
                q = make_query(rng, data)
                rids = balanced_candidates(rng, data, q)
                ids[b], lab[b], msk[b], _ = build_example(rng, grid, rids, q)
            logits = model.forward(ids, msk)
            loss, dlogits = binary_cross_entropy(logits, lab, msk)
            model.backward(dlogits)
            _accumulate_embed_grads(model, ids)
            _sgd_step(model, 0.02 * (0.97 ** (step / 300)))
            if step % 150 == 0 or step == STEPS - 1:
                print(f"    step {step:>4d}  loss {loss:.4f}")
        train_s = time.perf_counter() - t0
        print(f"  trained in {train_s:.1f}s  (~{sum(p.size for p in [model.token_embed, model.pos_embed]):,}+ params)")

        # ── 4. held-out queries: transformer vs exact SQL ────────────────
        TESTQ = 40
        tp = fp = fn = tn = 0
        t_tf = t_sql = 0.0
        for _ in range(TESTQ):
            q = make_query(rng, data)
            rids = balanced_candidates(rng, data, q)

            # transformer prediction (reads grid tokens)
            ids, _, msk, spans = build_example(rng, grid, rids, q)
            tt = time.perf_counter()
            logits = model.forward(ids[None, :], msk[None, :])[0]
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            t_tf += time.perf_counter() - tt
            tf_pred = []
            for s, e, _m in spans:
                seg = probs[s:min(e, SEQ)]
                tf_pred.append(1 if seg.mean() > 0.5 else 0)

            # sqlite exact ground truth (over the same K candidates)
            ts = time.perf_counter()
            rows = db.execute(
                f"SELECT id FROM recs WHERE id IN ({','.join('?'*K)}) "
                f"AND (abs(a-?)+abs(b-?)+abs(c-?)) < ?",
                (*[int(r) for r in rids], int(q[0]), int(q[1]), int(q[2]), THRESH),
            ).fetchall()
            t_sql += time.perf_counter() - ts
            sql_match = set(r[0] for r in rows)

            for rid, pred in zip(rids, tf_pred):
                truth = 1 if rid in sql_match else 0
                if pred and truth: tp += 1
                elif pred and not truth: fp += 1
                elif not pred and truth: fn += 1
                else: tn += 1

        total = tp + fp + fn + tn
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0

        # ── 5. determinism: same input -> identical logits ───────────────
        a1 = model.forward(ids[None, :], msk[None, :])
        a2 = model.forward(ids[None, :], msk[None, :])
        deterministic = np.array_equal(a1, a2)

        print()
        print("  ── RESULTS ──")
        print(f"  {'':20}{'5bit + transformer':>22}{'SQLite (exact)':>18}")
        print(f"  {'accuracy vs truth':20}{acc*100:>21.1f}%{'100.0% (truth)':>18}")
        print(f"  {'precision':20}{prec*100:>21.1f}%{'—':>18}")
        print(f"  {'recall':20}{rec*100:>21.1f}%{'—':>18}")
        print(f"  {'query latency':20}{t_tf/TESTQ*1e3:>20.2f}ms{t_sql/TESTQ*1e6:>15.1f}µs")
        print(f"  {'deterministic':20}{str(deterministic):>22}{'n/a':>18}")
        print()
        print(f"  confusion: tp={tp} fp={fp} fn={fn} tn={tn}  over {total} (query×candidate) pairs")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
