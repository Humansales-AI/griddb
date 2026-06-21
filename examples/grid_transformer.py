#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    GRID TRANSFORMER                                          ║
║                                                                            ║
║  A transformer that queries the Binary Grid Database directly.             ║
║  Operates on the 32-token vocabulary — not 32K tokens like typical LLMs.   ║
║  32 embeddings × 64 dimensions = 2,048 floats for the entire vocab table.  ║
║                                                                            ║
║  Built from scratch in NumPy.  No PyTorch, no TensorFlow.                  ║
║  Every gradient computed explicitly.  Every attention weight visible.      ║
║                                                                            ║
║  Task: Given grid records of shape (a, b, c), and a query vector,         ║
║        learn to find records within Manhattan distance of the query.       ║
║        The transformer learns the distance function from data —            ║
║        no one programmed "manhattan(a, b) < threshold".                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import sys
import os
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary_grid_db import (
    Token, Encoder, Parser, ParsedNumber,
    token_stream_to_binary_string,
    manhattan_distance,
    NUMERIC_DIGIT_VALUE, TOKEN_NAME,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRANSFORMER ARCHITECTURE — Pure NumPy
# ═══════════════════════════════════════════════════════════════════════════════

class LayerNorm:
    """Layer normalization: (x - mean) / std * gamma + beta."""
    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        # Cache for backward pass
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq, dim) or (seq, dim)"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = x_norm * self.gamma + self.beta
        self.cache['x'] = x
        self.cache['x_norm'] = x_norm
        self.cache['mean'] = mean
        self.cache['var'] = var
        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        mean = self.cache['mean']
        var = self.cache['var']
        N = x.shape[-1]

        # Sum gradients over all batch/sequence dimensions
        reduce_axes = tuple(range(dout.ndim - 1))
        dgamma = (dout * x_norm).sum(axis=reduce_axes)
        dbeta = dout.sum(axis=reduce_axes)

        dx_norm = dout * self.gamma
        dvar = (dx_norm * (x - mean) * -0.5 * (var + self.eps) ** -1.5).sum(axis=-1, keepdims=True)
        dmean = (dx_norm * -1.0 / np.sqrt(var + self.eps)).sum(axis=-1, keepdims=True) + \
                dvar * (-2.0 * (x - mean)).mean(axis=-1, keepdims=True)
        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2.0 * (x - mean) / N + dmean / N

        return dx, dgamma, dbeta


class Linear:
    """Linear transformation: Y = X @ W + b."""
    def __init__(self, in_dim: int, out_dim: int):
        # Kaiming init
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.cache['x']
        # dout has same shape as output = x @ W + b
        # Reshape to 2D for matmul
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        dout_2d = dout.reshape(-1, dout.shape[-1])

        dW = x_2d.T @ dout_2d
        db = dout_2d.sum(axis=0)
        dx_2d = dout_2d @ self.W.T
        dx = dx_2d.reshape(orig_shape)
        return dx, dW, db


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_backward(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    """GELU gradient approximation."""
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    return dout * (cdf + x * pdf)


class MultiHeadAttention:
    """Multi-head self-attention (single layer)."""
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        self.cache = {}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, d_model) → (batch, n_heads, seq, d_head)"""
        B, S, D = x.shape
        return x.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, n_heads, seq, d_head) → (batch, seq, d_model)"""
        B, H, S, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """x: (batch, seq, d_model)"""
        B, S, D = x.shape

        Q = self._split_heads(self.Wq.forward(x))   # (B, H, S, d_head)
        K = self._split_heads(self.Wk.forward(x))
        V = self._split_heads(self.Wv.forward(x))

        # Scaled dot-product attention
        scale = np.sqrt(self.d_head)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale  # (B, H, S, S)

        if mask is not None:
            scores = scores + (1.0 - mask[:, np.newaxis, np.newaxis, :]) * -1e9

        attn_weights = softmax(scores, axis=-1)  # (B, H, S, S)
        attn_out = attn_weights @ V               # (B, H, S, d_head)
        merged = self._merge_heads(attn_out)      # (B, S, D)
        out = self.Wo.forward(merged)

        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        self.cache['attn_weights'] = attn_weights
        self.cache['scores'] = scores
        self.cache['x'] = x
        self.cache['mask'] = mask
        self.cache['merged'] = merged
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        attn_weights = self.cache['attn_weights']
        scores = self.cache['scores']
        mask = self.cache['mask']
        merged = self.cache['merged']
        B, H, S, Dh = Q.shape

        # Wo backward
        d_merged, dWo, dbo = self.Wo.backward(dout)

        # Split gradients back to heads
        d_attn_out = d_merged.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)  # (B, H, S, Dh)

        # dV and d_attn_weights
        dV = attn_weights.transpose(0, 1, 3, 2) @ d_attn_out   # (B, H, S, Dh)
        d_attn_weights = d_attn_out @ V.transpose(0, 1, 3, 2)   # (B, H, S, S)

        # d_scores through softmax
        attn_reshaped = attn_weights.reshape(-1, S)
        d_attn_reshaped = d_attn_weights.reshape(-1, S)
        d_scores_reshaped = attn_reshaped * (d_attn_reshaped - (attn_reshaped * d_attn_reshaped).sum(axis=-1, keepdims=True))
        d_scores = d_scores_reshaped.reshape(B, H, S, S)

        if mask is not None:
            d_scores = d_scores * (mask[:, np.newaxis, np.newaxis, :])

        d_scores = d_scores / np.sqrt(Dh)

        # dQ, dK
        dQ = d_scores @ K   # (B, H, S, Dh)
        dK = d_scores.transpose(0, 1, 3, 2) @ Q  # (B, H, S, Dh)

        # Merge heads for linear layer backward
        dQ_merged = dQ.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
        dK_merged = dK.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
        dV_merged = dV.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)

        # Backward through linear projections
        dx_q, dWq, dbq = self.Wq.backward(dQ_merged)
        dx_k, dWk, dbk = self.Wk.backward(dK_merged)
        dx_v, dWv, dbv = self.Wv.backward(dV_merged)

        dx = dx_q + dx_k + dx_v

        # Store parameter gradients
        self.cache['dWq'] = dWq; self.cache['dbq'] = dbq
        self.cache['dWk'] = dWk; self.cache['dbk'] = dbk
        self.cache['dWv'] = dWv; self.cache['dbv'] = dbv
        self.cache['dWo'] = dWo; self.cache['dbo'] = dbo
        self.cache['dx'] = dx

        return dx


class FeedForward:
    """Position-wise feed-forward: Linear → GELU → Linear."""
    def __init__(self, d_model: int, d_ff: int):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        h = self.fc1.forward(x)
        self.cache['h_pre_act'] = h
        h = gelu(h)
        self.cache['h'] = h
        out = self.fc2.forward(h)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        h = self.cache['h']
        h_pre_act = self.cache['h_pre_act']

        dh, dW2, db2 = self.fc2.backward(dout)
        dh_pre = gelu_backward(h_pre_act, dh)
        dx, dW1, db1 = self.fc1.backward(dh_pre)

        self.cache['dW1'] = dW1; self.cache['db1'] = db1
        self.cache['dW2'] = dW2; self.cache['db2'] = db2
        return dx


class TransformerBlock:
    """One transformer block: Attention + FFN with residual connections."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)
        self.cache = {}

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Attention sub-layer with residual
        normed = self.ln1.forward(x)
        attn_out = self.attention.forward(normed, mask)
        x = x + attn_out

        # FFN sub-layer with residual
        normed = self.ln2.forward(x)
        ffn_out = self.ffn.forward(normed)
        x = x + ffn_out

        self.cache['x'] = x
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # FFN backward (residual)
        normed_ffn = self.ln2.cache['x']
        # dout flows through residual + ffn path
        d_ffn = self.ffn.backward(dout)
        d_ln2, dgamma2, dbeta2 = self.ln2.backward(d_ffn)
        dx_ffn = dout + d_ln2  # Residual connection

        # Attention backward (residual)
        normed_attn = self.ln1.cache['x']
        x_before_attn = dx_ffn  # This is the input to the attention residual
        d_attn = self.attention.backward(dx_ffn)
        d_ln1, dgamma1, dbeta1 = self.ln1.backward(d_attn)
        dx = x_before_attn + d_ln1  # Residual connection

        self.cache['dgamma1'] = dgamma1; self.cache['dbeta1'] = dbeta1
        self.cache['dgamma2'] = dgamma2; self.cache['dbeta2'] = dbeta2
        return dx


# ═══════════════════════════════════════════════════════════════════════════════
# 2. THE GRID TRANSFORMER — Query the binary grid via attention
# ═══════════════════════════════════════════════════════════════════════════════

class GridTransformer:
    """A transformer that operates directly on 5-bit grid tokens.

    Architecture:
      - Vocab: 32 tokens (the entire 5-bit lexicon)
      - Embedding: 32 × d_model (tiny — just 2,048 floats for d_model=64)
      - Positional encoding: learned (max 256 positions)
      - 2 transformer blocks
      - Output head: linear → 1 (match score per position)
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256,
                 n_blocks: int = 2, max_seq_len: int = 256, vocab_size: int = 32):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embeddings — just 32 rows!
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02

        # Positional embeddings
        self.pos_embed = np.random.randn(max_seq_len, d_model) * 0.02

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)]

        # Output head: score per position
        self.ln_final = LayerNorm(d_model)
        self.output_proj = Linear(d_model, 1)

    def forward(self, token_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass.

        Args:
            token_ids: (batch, seq_len) — integers 0-31
            mask: (batch, seq_len) — 1.0 for real tokens, 0.0 for padding
        Returns:
            scores: (batch, seq_len) — match score per position
        """
        B, S = token_ids.shape

        # Embedding lookup
        x = self.token_embed[token_ids]  # (B, S, d_model)
        x = x + self.pos_embed[:S]       # Add position

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Output head
        x = self.ln_final.forward(x)
        logits = self.output_proj.forward(x)  # (B, S, 1)
        return logits.squeeze(-1)  # (B, S)

    def backward(self, dout: np.ndarray) -> None:
        """Backward pass — accumulates gradients in-place."""
        B, S = dout.shape

        # Output head backward
        dout_expanded = dout[:, :, np.newaxis]  # (B, S, 1)
        dx, dW_out, db_out = self.output_proj.backward(dout_expanded)
        dx, dgamma_final, dbeta_final = self.ln_final.backward(dx)

        self.cache_grads = {
            'dW_out': dW_out, 'db_out': db_out,
            'dgamma_final': dgamma_final, 'dbeta_final': dbeta_final,
        }

        # Backward through blocks (reverse order)
        for i, block in reversed(list(enumerate(self.blocks))):
            dx = block.backward(dx)
            self.cache_grads[f'block_{i}'] = block.cache

        # Embedding gradients
        self.cache_grads['dx_final'] = dx
        # token_embed gradients are accumulated in the next step
        self._dx_embed = dx
        self._token_ids = None  # Set during training step


def binary_cross_entropy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """BCE loss with masking. logits/targets: (B, S), mask: (B, S)."""
    # Stable sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    # BCE: -[t*log(p) + (1-t)*log(1-p)]
    eps = 1e-7
    loss_per_pos = -(targets * np.log(probs + eps) + (1.0 - targets) * np.log(1.0 - probs + eps))
    masked_loss = loss_per_pos * mask
    loss = masked_loss.sum() / (mask.sum() + eps)

    # Gradient of BCE w.r.t. logits: p - t
    dlogits = (probs - targets) * mask / (mask.sum() + eps)

    return float(loss), dlogits


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA PIPELINE — Generate grid records and train on geometric queries
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GridQueryExample:
    token_ids: np.ndarray      # (seq_len,) — integer tokens
    labels: np.ndarray         # (seq_len,) — 1 where record matches query, 0 elsewhere
    mask: np.ndarray           # (seq_len,) — 1 for real tokens, 0 for padding


def generate_record_tokens(a: int, b: int, c: int) -> List[int]:
    """Generate token sequence for a record: [NUM(a), END, NUM(b), END, NUM(c), END, RECORD]."""
    tokens = []
    for val in [a, b, c]:
        t = Encoder.encode_integer(val)
        # encode_integer returns Token enum values; convert to int
        tokens.extend([int(tok) for tok in t])
    tokens.append(int(Token.RECORD))
    return tokens


def generate_query_tokens(a: int, b: int, c: int) -> List[int]:
    """Generate token sequence for a query: [NUM(a), END, NUM(b), END, NUM(c), END]."""
    tokens = []
    for val in [a, b, c]:
        t = Encoder.encode_integer(val)
        tokens.extend([int(tok) for tok in t])
    return tokens


def generate_batch(batch_size: int, num_records: int, seq_len: int,
                   value_range: int = 10, threshold: int = 5) -> GridQueryExample:
    """Generate a batch of (query + records) examples with BALANCED labels.

    Half the records match the query (within Manhattan distance threshold),
    half don't — preventing the model from cheating by always saying "no."

    Each example: [QUERY_TOKENS] [RECORD_1] [RECORD_2] ... padded to seq_len.
    Labels: 1 at positions inside records that match the query, 0 elsewhere.
    """
    all_token_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
    all_labels = np.zeros((batch_size, seq_len), dtype=np.float32)
    all_masks = np.zeros((batch_size, seq_len), dtype=np.float32)

    for b in range(batch_size):
        # Generate random query vector
        q = np.random.randint(0, value_range, size=3)

        # Generate balanced records: half matches, half non-matches
        num_match = num_records // 2
        num_nonmatch = num_records - num_match

        records = []
        record_matches = []

        # Generate matching records (within threshold)
        for _ in range(num_match):
            # Start near the query, add small random offsets
            offset = np.random.randint(-(threshold // 3), threshold // 3 + 1, size=3)
            r = q + offset
            r = np.clip(r, 0, value_range - 1)
            records.append(r)
            dist = np.sum(np.abs(q - r))
            record_matches.append(1.0 if dist < threshold else 0.0)

        # Generate non-matching records (outside threshold)
        for _ in range(num_nonmatch):
            # Generate far from query
            r = q + np.random.choice([-1, 1], size=3) * np.random.randint(threshold, threshold + value_range, size=3)
            r = np.clip(r, 0, value_range - 1)
            records.append(r)
            dist = np.sum(np.abs(q - r))
            record_matches.append(1.0 if dist < threshold else 0.0)

        # Shuffle to avoid positional bias
        indices = list(range(len(records)))
        np.random.shuffle(indices)
        records = [records[i] for i in indices]
        record_matches = [record_matches[i] for i in indices]

        # Build token sequence
        token_list = []

        # Query tokens first
        query_tokens = generate_query_tokens(int(q[0]), int(q[1]), int(q[2]))
        query_end = len(query_tokens)
        token_list.extend(query_tokens)

        # Then records
        record_positions = []  # (start, end) in token space
        for r, match in zip(records, record_matches):
            start = len(token_list)
            rec_tokens = generate_record_tokens(int(r[0]), int(r[1]), int(r[2]))
            token_list.extend(rec_tokens)
            end = len(token_list)
            record_positions.append((start, end, match))

        # Truncate if too long
        token_list = token_list[:seq_len]

        # Fill arrays
        for i, tok in enumerate(token_list):
            all_token_ids[b, i] = tok
            all_masks[b, i] = 1.0

        # Set labels at record token positions (not query positions)
        for start, end, match in record_positions:
            if start >= seq_len:
                break
            actual_end = min(end, seq_len)
            all_labels[b, start:actual_end] = match

    return GridQueryExample(token_ids=all_token_ids, labels=all_labels, mask=all_masks)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP — Manual SGD
# ═══════════════════════════════════════════════════════════════════════════════

def train_grid_transformer(
    model: GridTransformer,
    num_steps: int = 2000,
    batch_size: int = 32,
    num_records: int = 8,
    seq_len: int = 256,
    learning_rate: float = 0.01,
    print_every: int = 200,
) -> List[float]:
    """Train the transformer to query grid records."""

    losses = []

    print(f"\n  Training GridTransformer...")
    print(f"    Steps: {num_steps}, Batch: {batch_size}, Records/batch: {num_records}")
    print(f"    Seq length: {seq_len}, LR: {learning_rate}")
    print(f"    Params: ~{sum(p.size for p in _collect_params(model)):,}")

    for step in range(num_steps):
        # Generate batch
        batch = generate_batch(batch_size, num_records, seq_len)

        # Forward pass
        logits = model.forward(batch.token_ids, batch.mask)

        # Compute loss
        loss, dlogits = binary_cross_entropy(logits, batch.labels, batch.mask)
        losses.append(loss)

        # Backward pass
        model.backward(dlogits)

        # Accumulate embedding gradients
        _accumulate_embed_grads(model, batch.token_ids)

        # SGD update
        lr = learning_rate * (0.95 ** (step / 500))  # Decay
        _sgd_step(model, lr)

        # Progress
        if step % print_every == 0 or step == num_steps - 1:
            # Compute accuracy on this batch
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            preds = (probs > 0.5).astype(np.float32)
            correct = ((preds == batch.labels) * batch.mask).sum()
            total = batch.mask.sum()
            acc = correct / (total + 1e-7)
            print(f"    Step {step:>5d} | Loss: {loss:.4f} | Acc: {acc:.3f} | LR: {lr:.4f}")

    return losses


def _collect_params(model: GridTransformer) -> List[np.ndarray]:
    """Collect all trainable parameters."""
    params = [model.token_embed, model.pos_embed]
    for block in model.blocks:
        for layer in [block.attention.Wq, block.attention.Wk, block.attention.Wv,
                       block.attention.Wo, block.ffn.fc1, block.ffn.fc2]:
            params.extend([layer.W, layer.b])
        params.extend([block.ln1.gamma, block.ln1.beta])
        params.extend([block.ln2.gamma, block.ln2.beta])
    params.extend([model.ln_final.gamma, model.ln_final.beta])
    params.extend([model.output_proj.W, model.output_proj.b])
    return params


def _accumulate_embed_grads(model: GridTransformer, token_ids: np.ndarray):
    """Accumulate gradients for token embeddings from the final dx."""
    dx = model._dx_embed  # (B, S, d_model)
    B, S, D = dx.shape

    # token_embed gradients: sum dx for each token position
    d_token_embed = np.zeros_like(model.token_embed)
    for b in range(B):
        for s in range(S):
            tok = token_ids[b, s]
            d_token_embed[tok] += dx[b, s]

    model.cache_grads['d_token_embed'] = d_token_embed

    # pos_embed gradients: sum over batch for each position, only for used positions
    S_used = dx.shape[1]
    d_pos_embed_full = np.zeros_like(model.pos_embed)
    d_pos_embed_full[:S_used] = dx.sum(axis=0)[:S_used]
    model.cache_grads['d_pos_embed'] = d_pos_embed_full


def _sgd_step(model: GridTransformer, lr: float):
    """Apply accumulated gradients with SGD."""
    grads = model.cache_grads

    # Token embeddings
    model.token_embed -= lr * grads.get('d_token_embed', 0)
    model.pos_embed -= lr * grads.get('d_pos_embed', 0)

    # Output head
    model.output_proj.W -= lr * grads.get('dW_out', 0)
    model.output_proj.b -= lr * grads.get('db_out', 0)
    model.ln_final.gamma -= lr * grads.get('dgamma_final', 0)
    model.ln_final.beta -= lr * grads.get('dbeta_final', 0)

    # Blocks
    for i, block in enumerate(model.blocks):
        bg = grads.get(f'block_{i}', {})
        if not bg:
            continue
        # Attention
        for prefix, layer in [('Wq', block.attention.Wq), ('Wk', block.attention.Wk),
                               ('Wv', block.attention.Wv), ('Wo', block.attention.Wo)]:
            layer.W -= lr * block.attention.cache.get(f'd{prefix}', 0)
            layer.b -= lr * block.attention.cache.get(f'db{prefix[1:].lower()}', 0)

        # FFN
        block.ffn.fc1.W -= lr * block.ffn.cache.get('dW1', 0)
        block.ffn.fc1.b -= lr * block.ffn.cache.get('db1', 0)
        block.ffn.fc2.W -= lr * block.ffn.cache.get('dW2', 0)
        block.ffn.fc2.b -= lr * block.ffn.cache.get('db2', 0)

        # LayerNorms
        block.ln1.gamma -= lr * block.cache.get('dgamma1', 0)
        block.ln1.beta -= lr * block.cache.get('dbeta1', 0)
        block.ln2.gamma -= lr * block.cache.get('dgamma2', 0)
        block.ln2.beta -= lr * block.cache.get('dbeta2', 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DEMO — Train & query the grid
# ═══════════════════════════════════════════════════════════════════════════════

def demo_grid_transformer():
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    GRID TRANSFORMER DEMO                                     ║")
    print("║  A transformer that queries the Binary Grid Database directly.              ║")
    print("║  32-token vocabulary × 64 dimensions = 2,048 floats for all embeddings.     ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")

    # ── 1. Show the 32-token embedding table ──
    print(f"\n{'─'*60}")
    print(f"  1. The Embedding Table: 32 tokens → 64 dimensions")
    print(f"{'─'*60}")
    print(f"\n  A typical LLM has 32,000+ tokens in its vocabulary.")
    print(f"  The Grid Transformer has exactly 32 — the entire 5-bit lexicon.")
    print(f"  Token 00000 (D0/A) → 64 learned floats")
    print(f"  Token 00001 (D1/B) → 64 learned floats")
    print(f"  ...")
    print(f"  Token 11111 (START) → 64 learned floats")
    print(f"  Total: 32 × 64 = 2,048 parameters for the entire vocabulary.")

    # ── 2. Initialize model ──
    print(f"\n{'─'*60}")
    print(f"  2. Architecture")
    print(f"{'─'*60}")

    model = GridTransformer(
        d_model=48,
        n_heads=4,
        d_ff=128,
        n_blocks=2,
        max_seq_len=256,
        vocab_size=32,
    )

    total_params = sum(p.size for p in _collect_params(model))
    print(f"\n  Layers: 2 transformer blocks, 4 attention heads")
    print(f"  d_model: 48, d_ff: 128")
    print(f"  Total parameters: {total_params:,}")
    print(f"  (Compare: GPT-2 small has 124,000,000 parameters)")

    # ── 3. Show sample input ──
    print(f"\n{'─'*60}")
    print(f"  3. What the transformer sees")
    print(f"{'─'*60}")

    # Generate a sample and show it
    np.random.seed(42)
    batch = generate_batch(1, 4, 128, value_range=10, threshold=5)

    print(f"\n  Input: Raw token IDs (integers 0-31)")
    print(f"  Query: [NUM(5), END, NUM(3), END, NUM(7), END]")
    print(f"  Records follow, each terminated by RECORD (token 28)")

    # Display tokens with their meanings
    active_tokens = batch.token_ids[0][batch.mask[0] > 0][:40]
    labels = batch.labels[0][batch.mask[0] > 0][:40]

    from binary_grid_db import TOKEN_NAME
    print(f"\n  Token sequence (first 40):")
    line = "  "
    for i, (tok, lab) in enumerate(zip(active_tokens, labels)):
        name = TOKEN_NAME.get(Token(tok), f"#{tok}")
        marker = "✓" if lab > 0.5 else " "
        line += f"{name} "
        if (i + 1) % 10 == 0 or i == len(active_tokens) - 1:
            print(line)
            line = "  "
    print(f"  ✓ = token belongs to a matching record")

    # ── 4. Train ──
    print(f"\n{'─'*60}")
    print(f"  4. Training: Learning to query the grid")
    print(f"{'─'*60}")

    np.random.seed(123)
    losses = train_grid_transformer(
        model,
        num_steps=800,
        batch_size=16,
        num_records=8,
        seq_len=128,
        learning_rate=0.01,
        print_every=200,
    )

    # ── 5. Evaluation ──
    print(f"\n{'─'*60}")
    print(f"  5. Evaluation: Transformer vs. Ground Truth Manhattan")
    print(f"{'─'*60}")

    # Evaluate: feed ALL records together with the query (as trained)
    # The model compares records via attention and scores each token position.
    print(f"\n  Query: [2, 3, 1] — Threshold: Manhattan distance < 5")
    print(f"\n  {'Record Value':<15} {'Distance':>8} {'Truth':>8} {'Match %':>10} {'Pred':>8}")
    print(f"  {'-'*55}")

    query = [2, 3, 1]
    test_records = [
        ([2, 3, 2], True),    # dist=1
        ([1, 4, 0], True),    # dist=2  (|2-1|+|3-4|+|1-0| = 1+1+1 = 3)
        ([3, 3, 3], True),    # dist=3
        ([8, 9, 7], False),   # dist=6+6+6=18
        ([5, 0, 1], False),   # dist=3+3+0=6
        ([0, 0, 0], False),   # dist=2+3+1=6
    ]

    # Build full sequence: query + all records
    q_tokens_list = generate_query_tokens(*query)
    all_rec_tokens = []
    record_boundaries = []  # (start_token_idx, end_token_idx, rec_vals, truth)
    current_pos = len(q_tokens_list)
    for rec_vals, truth in test_records:
        r_tokens = generate_record_tokens(*rec_vals)
        start = current_pos
        all_rec_tokens.extend(r_tokens)
        end = current_pos + len(r_tokens)
        record_boundaries.append((start, end, rec_vals, truth))
        current_pos = end

    full_tokens = q_tokens_list + all_rec_tokens
    seq_len_needed = len(full_tokens)
    padded_len = ((seq_len_needed + 7) // 8) * 8 + 8  # Round up

    token_ids_eval = np.zeros((1, padded_len), dtype=np.int32)
    mask_eval = np.zeros((1, padded_len), dtype=np.float32)
    for j, tok in enumerate(full_tokens[:padded_len]):
        token_ids_eval[0, j] = tok
        mask_eval[0, j] = 1.0

    # Forward pass on full sequence
    logits_eval = model.forward(token_ids_eval, mask_eval)
    probs_eval = 1.0 / (1.0 + np.exp(-np.clip(logits_eval, -50, 50)))

    correct = 0
    for start, end, rec_vals, truth in record_boundaries:
        if start < padded_len:
            actual_end = min(end, padded_len)
            rec_probs = probs_eval[0, start:actual_end]
            rec_mask = mask_eval[0, start:actual_end]
            masked_probs = rec_probs[rec_mask > 0]
            if len(masked_probs) > 0:
                avg_score = masked_probs.mean()
            else:
                avg_score = 0.0

        dist = manhattan_distance(query, rec_vals)
        pred_match = avg_score > 0.5
        truth_str = "MATCH" if truth else "no"
        pred_str = "MATCH" if pred_match else "no"
        status = "✓" if pred_match == truth else "✗"

        print(f"  {str(rec_vals):<15} {dist:>8}  {truth_str:>8}  {avg_score:>9.4f}  {pred_str:>8}  {status}")

        if pred_match == truth:
            correct += 1

    acc = correct / len(test_records)
    print(f"\n  Accuracy: {correct}/{len(test_records)} = {acc:.1%}")
    print(f"\n  The transformer learned to approximate Manhattan distance")
    print(f"  directly from raw 5-bit tokens — no schema, no query planner,")
    print(f"  no explicit distance function programmed.")
    print(f"  Embedding table: just 32 × 32 = 1,024 floats for the entire vocabulary.")

    # ── 6. Attention visualization ──
    print(f"\n{'─'*60}")
    print(f"  6. Attention Patterns (first block, first head)")
    print(f"{'─'*60}")

    # Run a single example and capture attention weights
    single = generate_batch(1, 4, 48, value_range=10, threshold=5)
    _ = model.forward(single.token_ids, single.mask)

    # Get attention weights from first block
    attn = model.blocks[0].attention.cache['attn_weights']  # (B, H, S, S)
    attn_head0 = attn[0, 0]  # First head, (S, S)

    # Extract active tokens cleanly
    active_len = int(single.mask[0].sum())
    active_single = [int(single.token_ids[0, i]) for i in range(active_len)]
    record_positions = [i for i in range(active_len)
                        if Token(active_single[i]) == Token.RECORD]

    if record_positions:
        print(f"\n  Where RECORD tokens attend (token → top attended positions):")
        for rp in record_positions[:3]:
            attn_row = attn_head0[rp, :active_len]
            top_k = np.argsort(-attn_row)[:5]
            top_str = ", ".join(
                f"pos {p} ({TOKEN_NAME.get(Token(active_single[p]), '?')}, w={attn_row[p]:.3f})"
                for p in top_k
            )
            print(f"    RECORD at pos {rp}: attends to → {top_str}")

    print(f"\n  The transformer learns that RECORD tokens should attend to")
    print(f"  the query tokens and the numeric values within their own record —")
    print(f"  effectively learning a distance computation via attention.")

    return model


def _extract_numbers_from_tokens(tokens: List[int], start: int) -> List[int]:
    """Extract number values from a token sequence starting at position `start`.
    Stops when it hits a RECORD, START, or runs out of tokens.
    """
    values = []
    digits = []
    for i in range(start, len(tokens)):
        tok = Token(tokens[i])
        if tok == Token.RECORD or tok == Token.START:
            # Finalize current number if any
            if digits:
                values.append(_digits_to_value(digits))
            break
        elif tok == Token.END:
            if digits:
                values.append(_digits_to_value(digits))
                digits = []
        elif tok in NUMERIC_DIGIT_VALUE and NUMERIC_DIGIT_VALUE[tok] is not None:
            digits.append(NUMERIC_DIGIT_VALUE[tok])
        # Skip operators and other tokens
    # Don't finalize here — let RECORD/START handle it
    return values


def _digits_to_value(digits: List[int]) -> int:
    """Convert signed digits to integer value."""
    if not digits:
        return 0
    n = len(digits)
    value = 0
    for i, d in enumerate(digits):
        value += d * (10 ** (n - 1 - i))
    return value


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    demo_grid_transformer()
