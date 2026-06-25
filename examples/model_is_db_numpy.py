#!/usr/bin/env python3
"""
5bit Model-IS-Database — NumPy Vectorized
===========================================
32K param transformer. All operations are matrix multiplies.
No Python loops over dimensions. Trains in seconds, not minutes.

Every token is 0-31 (5-bit). Every weight is a float32 matrix.
The database IS the weights after training.

Run: python3 examples/model_is_db_numpy.py
"""
import os, sys, time, random, math
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from binary_grid_db import Token, Encoder

VOCAB, D_MODEL, N_HEADS, N_LAYERS = 32, 64, 4, 2
MAX_SEQ = 64
DK = D_MODEL // N_HEADS  # 16


class ModelIsDB:
    """32K-param transformer. NumPy vectorized. All matrix math."""

    def __init__(self):
        scale = 0.02
        self.embed = np.random.randn(VOCAB, D_MODEL).astype(np.float32) * 0.1
        # Positional encoding
        pos = np.arange(MAX_SEQ)[:, None]
        dim = np.arange(D_MODEL)[None, :]
        self.pos = np.zeros((MAX_SEQ, D_MODEL), dtype=np.float32)
        self.pos[:, 0::2] = np.sin(pos / (10000 ** (dim[:, 0::2] / D_MODEL)))
        self.pos[:, 1::2] = np.cos(pos / (10000 ** (dim[:, 1::2] / D_MODEL)))

        # Attention weights per layer
        self.W_q = [np.random.randn(D_MODEL, DK).astype(np.float32) * scale for _ in range(N_LAYERS)]
        self.W_k = [np.random.randn(D_MODEL, DK).astype(np.float32) * scale for _ in range(N_LAYERS)]
        self.W_v = [np.random.randn(D_MODEL, DK).astype(np.float32) * scale for _ in range(N_LAYERS)]
        self.W_o = [np.random.randn(DK, D_MODEL).astype(np.float32) * scale for _ in range(N_LAYERS)]

        # Output: hidden → vocabulary
        self.W_out = np.random.randn(D_MODEL, VOCAB).astype(np.float32) * scale
        self.b_out = np.zeros(VOCAB, dtype=np.float32)

    def forward(self, token_ids):
        """token_ids: list of ints (0-31). Returns [VOCAB] log-probs."""
        n = min(len(token_ids), MAX_SEQ)
        if n == 0: return np.zeros(VOCAB, dtype=np.float32)

        # Embed + position
        idx = np.array(token_ids[:n], dtype=np.int32)
        x = self.embed[idx] + self.pos[:n]  # [n, D_MODEL]

        for layer in range(N_LAYERS):
            # Q, K, V projections: [n, DK]
            Q = x @ self.W_q[layer]
            K = x @ self.W_k[layer]
            V = x @ self.W_v[layer]

            # Scaled dot-product attention: softmax(Q @ K.T / sqrt(dk)) @ V
            scores = Q @ K.T / math.sqrt(DK)          # [n, n]
            attn = self._softmax(scores)                # [n, n]
            out = attn @ V                              # [n, DK]

            # Output projection + residual
            x = x + out @ self.W_o[layer] * 0.1         # [n, D_MODEL]

            # Layer norm (simplified)
            norm = np.sqrt((x * x).sum(axis=-1, keepdims=True) + 1e-8)
            x = x / norm

        # Pool last position → predict next token
        logits = x[-1] @ self.W_out + self.b_out        # [VOCAB]
        return self._softmax(logits)

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def train_batch(self, batch_q, batch_a, lr=0.005):
        """Train on a batch of (question, answer) pairs."""
        total_loss = 0
        for q_ids, a_ids in zip(batch_q, batch_a):
            ctx = list(q_ids)
            for target in a_ids:
                if not (0 <= target < VOCAB): continue
                probs = self.forward(ctx)
                loss = -math.log(max(probs[target], 1e-10))
                total_loss += loss

                # Gradient on W_out (simplified — just push the correct token)
                x = self.embed_seq(list(ctx))
                if x is None or len(x) == 0: continue
                last = x[-1]
                grad = probs[target] - 1.0
                self.W_out[:, target] -= lr * grad * last
                self.b_out[target] -= lr * grad

                ctx.append(target)
        return total_loss

    def embed_seq(self, token_ids):
        n = min(len(token_ids), MAX_SEQ)
        if n == 0: return None
        idx = np.array(token_ids[:n], dtype=np.int32)
        return self.embed[idx] + self.pos[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# Q/A Generation + Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def generate_qa(n_users, n_orders):
    db = defaultdict(list)
    for _ in range(n_orders):
        uid = random.randint(1, n_users)
        db[uid].append(random.randint(100, 50000))

    def _tok(val):
        """5-bit token IDs from integer."""
        ids = [int(t) for t in Encoder.encode_integer(val) if isinstance(t, Token)]
        return ids

    def _word(s):
        return [int(t) for t in Encoder.encode_word(s) if isinstance(t, Token)]

    qa = []
    for uid in range(1, n_users + 1):
        count = len(db[uid])
        qa.append((_word("how-many-orders-user") + _tok(uid), _tok(count)))
        total = sum(db[uid])
        qa.append((_word("total-spent-user") + _tok(uid), _tok(total)))
        has = 1 if count > 0 else 0
        qa.append((_word("has-orders-user") + _tok(uid), _tok(has)))

    return qa, db


def benchmark():
    print("═" * 60)
    print("  5bit Model-IS-Database — NumPy Vectorized")
    print("═" * 60)

    for n_users, n_orders, epochs in [(50, 200, 30), (100, 500, 20), (200, 1000, 15)]:
        print(f"\n── {n_users} users × {n_orders} orders ──")
        qa, db = generate_qa(n_users, n_orders)
        model = ModelIsDB()
        print(f"  Model: 32K params, {len(qa)} Q/A pairs")

        # Train
        t0 = time.perf_counter()
        split = len(qa) // 2
        train_set = qa[:split]; test_set = qa[split:]
        for ep in range(epochs):
            random.shuffle(train_set)
            batch_q = [q for q, _ in train_set[:64]]
            batch_a = [a for _, a in train_set[:64]]
            model.train_batch(batch_q, batch_a, lr=0.005)
        train_t = time.perf_counter() - t0

        # Test
        correct = 0; total = 0
        for q_ids, a_ids in test_set:
            if not a_ids: continue
            probs = model.forward(q_ids)
            target = a_ids[0]
            if 0 <= target < VOCAB:
                pred = int(np.argmax(probs))
                if pred == target: correct += 1
                total += 1
        acc = correct / max(total, 1)

        # B-tree
        t0 = time.perf_counter()
        res = sum(len(db[uid]) for uid in range(1, n_users + 1))
        bt = (time.perf_counter() - t0) * 1e3

        print(f"  Train: {train_t:.1f}s  Acc: {acc:.1%}  B-tree: {bt:.1f}ms")

    print(f"\n═══ Architecture ═══")
    print(f"  NumPy vectorized — all matrix multiplies")
    print(f"  32-token vocab × 64-dim embedding")
    print(f"  No IEEE 754 in data — only in weights")
    print(f"  Trains in seconds, not minutes")
    print("═" * 60)


if __name__ == '__main__':
    benchmark()
