#!/usr/bin/env python3
"""
5bit Model-IS-Database — NumPy + Attention Backprop
=====================================================
Attention layer with full gradient computation.
No framework — just numpy + known gradient formulas.

Softmax attention backprop:
  d_scores = attn * (d_attn - sum(attn * d_attn, axis=-1)) / sqrt(dk)
  d_Q = d_scores @ K
  d_K = d_scores^T @ Q

Run: python3 examples/model_is_db_backprop.py
"""
import os, sys, time, random, math
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from binary_grid_db import Token, Encoder

VOCAB, D_MODEL, MAX_SEQ = 32, 32, 32


class AttentionModel:
    """Single attention layer + output. Full gradient backprop. 5-bit vocab."""

    def __init__(self):
        d = 0.02
        self.embed = np.random.randn(VOCAB, D_MODEL).astype(np.float32) * 0.1
        self.W_q = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * d
        self.W_k = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * d
        self.W_v = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * d
        self.W_o = np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * d
        self.W_out = np.random.randn(D_MODEL, VOCAB).astype(np.float32) * d
        self.b_out = np.zeros(VOCAB, dtype=np.float32)

    def forward(self, token_ids):
        """Returns probs [VOCAB] and cache dict for backward."""
        n = min(len(token_ids), MAX_SEQ)
        if n == 0:
            return np.ones(VOCAB, dtype=np.float32) / VOCAB, {}

        idx = np.array(token_ids[:n], dtype=np.int32)
        x = self.embed[idx]  # [n, D]

        # Attention
        Q = x @ self.W_q; K = x @ self.W_k; V = x @ self.W_v  # [n, D]
        dk = math.sqrt(D_MODEL)
        scores = Q @ K.T / dk                                   # [n, n]
        attn = self._softmax(scores)                            # [n, n]
        out = attn @ V                                          # [n, D]
        h = x + out @ self.W_o * 0.1                            # residual [n, D]

        # Pool last position → output
        logits = h[-1] @ self.W_out + self.b_out                # [VOCAB]
        probs = self._softmax(logits[None, :])[0]

        cache = {'x': x, 'Q': Q, 'K': K, 'V': V, 'scores': scores, 'attn': attn, 'out': out, 'h': h, 'idx': idx}
        return probs, cache

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def train_step(self, q_ids, a_ids, lr=0.01):
        loss = 0.0
        ctx = list(q_ids)
        for target in a_ids:
            if not (0 <= target < VOCAB): continue

            probs, c = self.forward(ctx)
            if not c: continue
            p = max(probs[target], 1e-10)
            loss -= math.log(p)

            # ── Output gradients ──────────────────────────────────
            d_logits = probs.copy()
            d_logits[target] -= 1.0
            d_h_last = self.W_out @ d_logits  # [D]

            self.W_out -= lr * np.outer(c['h'][-1], d_logits)
            self.b_out -= lr * d_logits

            # ── Attention gradients ───────────────────────────────
            # dL/d_out = d_h_last * 0.1 (residual scale)
            d_out = np.zeros_like(c['out'])
            d_out[-1] = d_h_last * 0.1

            # dV = attn^T @ d_out, d_attn = d_out @ V^T
            d_V = c['attn'].T @ d_out
            d_attn = d_out @ c['V'].T

            # Softmax backward
            d_scores = c['attn'] * (d_attn - (c['attn'] * d_attn).sum(axis=-1, keepdims=True)) / math.sqrt(D_MODEL)

            # Q, K, V projection gradients
            d_Q = d_scores @ c['K']
            d_K = d_scores.T @ c['Q']

            # Weight gradients
            self.W_q -= lr * 0.1 * c['x'].T @ d_Q / c['x'].shape[0]
            self.W_k -= lr * 0.1 * c['x'].T @ d_K / c['x'].shape[0]
            self.W_v -= lr * 0.1 * c['x'].T @ d_V / c['x'].shape[0]
            self.W_o -= lr * 0.01 * c['out'].T @ d_out / c['x'].shape[0]

            # Embedding gradient
            d_x = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T
            n = c['idx'].shape[0]
            for i in range(n):
                self.embed[c['idx'][i]] -= lr * 0.01 * d_x[i] / n

            ctx.append(target)
        return loss

    def predict(self, q_ids):
        probs, _ = self.forward(q_ids)
        return int(np.argmax(probs))


def generate_qa(n_users, n_orders):
    db = defaultdict(list)
    for _ in range(n_orders):
        uid = random.randint(1, n_users)
        db[uid].append(random.randint(100, 50000))

    def _tok(val):
        return [int(t) for t in Encoder.encode_integer(val) if isinstance(t, Token)]

    def _word(s):
        return [int(t) for t in Encoder.encode_word(s) if isinstance(t, Token)]

    qa = []
    for uid in range(1, n_users + 1):
        count = len(db[uid])
        qa.append((_word("count") + _tok(uid), _tok(count)))
        total = sum(db[uid])
        qa.append((_word("total") + _tok(uid), _tok(total)))

    return qa, db


def benchmark():
    print("═" * 60)
    print("  5bit Model-IS-Database — NumPy + Full Backprop")
    print("═" * 60)

    for n_users, n_orders, epochs in [(50, 200, 150), (100, 500, 120), (200, 1000, 100)]:
        print(f"\n── {n_users} users × {n_orders} orders ──")
        qa, db = generate_qa(n_users, n_orders)
        model = AttentionModel()
        params = VOCAB * D_MODEL + D_MODEL * D_MODEL * 5 + D_MODEL * VOCAB
        print(f"  Model: {params:,} params, {len(qa)} Q/A pairs")

        split = len(qa) // 2
        t0 = time.perf_counter()
        for ep in range(epochs):
            random.shuffle(qa)
            batch_loss = 0
            for q, a in qa[:split]:
                batch_loss += model.train_step(q, a, lr=0.005)
        train_t = time.perf_counter() - t0

        # Test
        correct = 0; total = 0
        for q, a in qa[split:]:
            if not a: continue
            pred = model.predict(q)
            if pred == a[0]: correct += 1
            total += 1
        acc = correct / max(total, 1)

        t0 = time.perf_counter()
        res = sum(len(db[uid]) for uid in db)
        bt = (time.perf_counter() - t0) * 1e3

        print(f"  Train: {train_t:.1f}s  Acc: {acc:.1%}  B-tree: {bt:.1f}ms")

    print(f"\n═══ Two-layer MLP + full backprop ═══")
    print(f"  Embedding → ReLU → Softmax")
    print(f"  5-bit tokens in, predictions out")
    print("═" * 60)


if __name__ == '__main__':
    benchmark()
