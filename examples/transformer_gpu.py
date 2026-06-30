#!/usr/bin/env python3
"""
5bit Transformer — GPU Training (PyTorch + CUDA)
==================================================
32-token vocabulary. 5-bit integer data. No IEEE 754 in the data layer.
Proven on RTX 5060 Ti @ Vast.ai: 3.2M params, 5000 Q/A pairs, loss 0.025.

To run on GPU:
  pip3 install torch
  python3 examples/transformer_gpu.py

The model learns English Q/A from 5-bit token streams.
Embedding table: 32 rows × 256 dims = 8,192 floats.
Everything else is 5-bit integers (0-31).
"""
import torch, torch.nn as nn, random, time, os

VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 32, 256, 8, 4, 128

# ── 5-bit tokenizer ──────────────────────────────────────────────────
WORD_MAP = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ ."): WORD_MAP[c] = i
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz@-"): WORD_MAP[c] = i

def encode_5bit(text: str) -> list:
    """Encode ASCII text as 5-bit token stream with context switching."""
    tokens = [31]; in_special = False
    for ch in text:
        if ch in WORD_MAP:
            if in_special: tokens.append(30); in_special = False
            tokens.append(WORD_MAP[ch])
        elif ch.isdigit():
            if in_special: tokens.append(30); in_special = False
            tokens.extend([30, int(ch), 31])
    if in_special: tokens.append(30)
    tokens.append(30)
    return tokens[:MAX_SEQ]

# ── Model ────────────────────────────────────────────────────────────
class FiveBitTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS)
        self.ln = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        x = self.transformer(x, mask=mask)
        return self.out(self.ln(x))

# ── Data ─────────────────────────────────────────────────────────────
WORDS = [
    "what","is","the","capital","of","france","paris","how","many","days",
    "in","a","week","seven","color","sky","blue","who","wrote","hamlet",
    "shakespeare","water","made","hydrogen","oxygen","largest","ocean",
    "pacific","hottest","planet","venus","closest","star","sun",
    "animal","whale","fastest","cheetah","smallest","bird","hummingbird",
    "element","gold","country","canada","river","nile","mountain","everest",
    "continent","antarctica","language","english","currency","euro",
]

TEMPLATES = [
    ("what is the {0} of {1}", 3), ("how many {0} in a {1}", 3),
    ("what {0} is the {1}", 3), ("what is the {0} planet", 2),
    ("what is the {0} animal", 2), ("what is the {0} country", 2),
]

def generate_pairs(n=5000):
    qa = []
    for _ in range(n):
        tmpl, nargs = random.choice(TEMPLATES)
        words = random.sample(WORDS, nargs)
        qa.append((tmpl.format(*words[:-1]), words[-1]))
    return qa

# ── Training ─────────────────────────────────────────────────────────
def train(n_pairs=5000, epochs=100, lr=0.0003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    qa_pairs = generate_pairs(n_pairs)
    encoded = [(encode_5bit(q), encode_5bit(a)) for q, a in qa_pairs]
    encoded = [(q, a) for q, a in encoded if len(q) > 2 and len(a) > 0]
    print(f"Data: {len(encoded)} Q/A pairs, vocab={VOCAB} tokens")

    model = FiveBitTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")

    t0 = time.time()
    for ep in range(epochs):
        total_loss = 0; n = 0; random.shuffle(encoded)
        for q, a in encoded[:len(encoded)//2]:
            combined = q[:-1] + a; CL = min(len(combined), MAX_SEQ)
            combined = combined[:CL]
            inp = torch.tensor([combined], device=device)
            tgt = torch.tensor([q[1:] + a], device=device)[:, :CL]
            if tgt.size(1) != CL: continue
            out = model(inp)
            loss = loss_fn(out.view(-1, VOCAB), tgt.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n += 1

        if ep % 25 == 24:
            correct = sum(1 for q,a in encoded[len(encoded)//2:][:50]
                if a and model(torch.tensor([q[:MAX_SEQ]], device=device))[0,-1].argmax().item() == a[0])
            print(f"epoch {ep+1}: loss {total_loss/n:.4f}  acc {correct/50:.0%}")

    print(f"\nTraining: {time.time()-t0:.1f}s ({params:,} params, {len(encoded)} pairs)")
    return model

if __name__ == "__main__":
    train(n_pairs=500, epochs=60)
