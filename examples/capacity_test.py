#!/usr/bin/env python3
"""
Capacity test: same SMALL model, growing dataset -> where does exact-match break?

Exact memorization stores data IN the weights. A fixed-size model has finite
capacity, so as the closed dataset grows, exact-match should eventually fall
below 100%. This finds that wall empirically instead of guessing.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python")); sys.path.insert(0, HERE)
import nl_query as nlq
from grid_transformer import _collect_params

VR = nlq.VRANGE


def run(Nrec, steps=800, batch=128):
    rng = np.random.RandomState(7)
    data = rng.randint(0, VR, size=(Nrec, 3))
    Q, Y = [], []
    for ai in range(3):
        for rid in range(Nrec):
            Q.append(nlq.encode_question(ai, rid)); Y.append(int(data[rid, ai]))
    ids = np.zeros((len(Q), nlq.SEQ), dtype=np.int32)
    msk = np.zeros((len(Q), nlq.SEQ), dtype=np.float32)
    for i, q in enumerate(Q):
        ids[i, :len(q)] = q; msk[i, :len(q)] = 1.0
    Y = np.array(Y)

    np.random.seed(7)
    model = nlq.GridQA(n_classes=VR)
    nparam = sum(p.size for p in _collect_params(model.m))
    t0 = time.perf_counter()
    for step in range(steps):
        idx = rng.randint(0, len(Q), size=batch)
        model.step(ids[idx], msk[idx], Y[idx], 0.03 * (0.97 ** (step / 400)))
    dt = time.perf_counter() - t0
    preds = model.forward(ids, msk).argmax(axis=1)
    exact = (preds == Y).mean()
    return nparam, len(Q), exact, dt


def main():
    print(f"  small model — same params each row")
    print(f"  {'records':>8}{'Q/A pairs':>11}{'params':>9}{'exact-match':>13}{'train s':>9}")
    for Nrec in [64, 2048]:
        nparam, npairs, exact, dt = run(Nrec)
        print(f"  {Nrec:>8}{npairs:>11}{nparam:>9,}{exact*100:>12.1f}%{dt:>8.0f}s")


if __name__ == "__main__":
    main()
