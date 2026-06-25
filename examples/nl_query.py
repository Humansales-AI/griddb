#!/usr/bin/env python3
"""
NL-ish query -> EXACT answer over a CLOSED 5bit dataset.

The point (the user's actual idea):
  A tiny deterministic transformer that knows ONLY this dataset + a little
  grammar. You ask "value of B for record 12" in the 5bit lexicon; it returns
  the EXACT stored value via argmax (a hard discrete output, not a probability).
  Memorization is correct here: you only ever query data that EXISTS — exactly
  like SQLite already has the rows. Closed world, both.

We train on the full closed Q/A universe, then check EXACT-MATCH on it, plus
determinism (same input -> identical argmax). SQLite would need exact SQL
(`SELECT b FROM r WHERE id=12`); here the query is just tokens + English.

Reuses the tested transformer components; classification head + mean-pool added.
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, HERE)

from binary_grid_db import Encoder, Token                       # noqa: E402
from grid_transformer import (                                  # noqa: E402
    GridTransformer, TransformerBlock, Linear, LayerNorm,
    _sgd_step, _accumulate_embed_grads,
)

SEED = 42
N = 32           # records
VRANGE = 10      # attribute values 0..9  -> answer classes
ATTRS = ["A", "B", "C"]
SEQ = 24
STEPS = 1500


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def encode_question(attr_idx, rid):
    """'<attr> OF RECORD <id>' in the 5bit lexicon: word tokens + integer id."""
    toks = list(Encoder.encode_word(ATTRS[attr_idx]))
    toks += list(Encoder.encode_word("OF"))
    toks += list(Encoder.encode_word("RECORD"))
    toks += list(Encoder.encode_integer(int(rid)))
    return [int(t) for t in toks]


class GridQA:
    """Transformer trunk (reused) + mean-pool + classification head."""
    def __init__(self, n_classes):
        self.m = GridTransformer(d_model=64, n_heads=4, d_ff=128, n_blocks=2, max_seq_len=SEQ)
        self.m.output_proj = Linear(64, n_classes)   # _sgd_step updates this as the head
        self.m.ln_final = LayerNorm(64)

    def forward(self, ids, mask):
        m = self.m
        x = m.token_embed[ids] + m.pos_embed[:ids.shape[1]]
        for blk in m.blocks:
            x = blk.forward(x, mask)
        self._x = x
        self._mask = mask
        cnt = mask.sum(axis=1, keepdims=True) + 1e-7
        pooled = (x * mask[:, :, None]).sum(axis=1) / cnt        # (B, d)
        h = m.ln_final.forward(pooled)
        logits = m.output_proj.forward(h)                        # (B, C)
        return logits

    def step(self, ids, mask, y, lr):
        m = self.m
        logits = self.forward(ids, mask)
        probs = softmax_rows(logits)
        B = ids.shape[0]
        loss = -np.log(probs[np.arange(B), y] + 1e-9).mean()
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B

        m.cache_grads = {}
        dh, dW_out, db_out = m.output_proj.backward(dlogits)
        dpooled, dgf, dbf = m.ln_final.backward(dh)
        m.cache_grads.update(dW_out=dW_out, db_out=db_out,
                             dgamma_final=dgf, dbeta_final=dbf)
        cnt = mask.sum(axis=1, keepdims=True) + 1e-7
        dx = (dpooled / cnt)[:, None, :] * mask[:, :, None]      # scatter pool grad
        for blk in reversed(m.blocks):
            dx = blk.backward(dx)
        m._dx_embed = dx
        _accumulate_embed_grads(m, ids)
        _sgd_step(m, lr)
        return float(loss)

    def predict(self, ids, mask):
        return int(self.forward(ids, mask).argmax(axis=1)[0])


def main():
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    data = rng.randint(0, VRANGE, size=(N, 3))   # the closed dataset

    # full closed Q/A universe: every (attr, record) -> its exact value
    Q, Y = [], []
    for ai in range(3):
        for rid in range(N):
            Q.append(encode_question(ai, rid))
            Y.append(int(data[rid, ai]))
    maxlen = max(len(q) for q in Q)
    assert maxlen <= SEQ, maxlen
    ids = np.zeros((len(Q), SEQ), dtype=np.int32)
    msk = np.zeros((len(Q), SEQ), dtype=np.float32)
    for i, q in enumerate(Q):
        ids[i, :len(q)] = q
        msk[i, :len(q)] = 1.0
    Y = np.array(Y)
    print(f"  closed dataset: {N} records, {len(Q)} Q/A pairs (every attr x record)")

    model = GridQA(n_classes=VRANGE)
    t0 = time.perf_counter()
    for step in range(STEPS):
        perm = rng.permutation(len(Q))
        loss = model.step(ids[perm], msk[perm], Y[perm], 0.03 * (0.97 ** (step / 300)))
        if step % 200 == 0 or step == STEPS - 1:
            preds = model.forward(ids, msk).argmax(axis=1)
            acc = (preds == Y).mean()
            print(f"    step {step:>4d}  loss {loss:.4f}  exact-match {acc*100:5.1f}%")
    train_s = time.perf_counter() - t0

    # final exact-match over the whole closed universe
    preds = model.forward(ids, msk).argmax(axis=1)
    exact = (preds == Y).mean()

    # determinism: same query -> identical argmax, twice
    det = all(model.predict(ids[i:i+1], msk[i:i+1]) == model.predict(ids[i:i+1], msk[i:i+1])
              for i in range(len(Q)))

    # latency for one query
    t = time.perf_counter()
    for _ in range(200):
        model.predict(ids[5:6], msk[5:6])
    q_us = (time.perf_counter() - t) / 200 * 1e6

    # show a couple of real answers
    print(f"\n  trained in {train_s:.1f}s")
    print(f"  EXACT-MATCH on closed set : {exact*100:.1f}%   ({(preds==Y).sum()}/{len(Q)})")
    print(f"  deterministic              : {det}")
    print(f"  query latency              : {q_us:.0f} µs")
    print(f"\n  sample — ask in tokens, get exact stored value:")
    for ai, rid in [(0, 12), (1, 12), (2, 7), (1, 30)]:
        i = ai * N + rid
        p = model.predict(ids[i:i+1], msk[i:i+1])
        ok = "✓" if p == data[rid, ai] else "✗"
        print(f"    '{ATTRS[ai]} OF RECORD {rid}'  -> {p}   (stored {data[rid,ai]}) {ok}")


if __name__ == "__main__":
    main()
