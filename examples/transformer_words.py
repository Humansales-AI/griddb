#!/usr/bin/env python3
"""5bit Word-Level Transformer — 50K words as 5-bit letter sequences."""
import torch, torch.nn as nn, random, json, time

VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 32, 256, 8, 4, 256
SPACE = 26  # Token 26 = space between words

# Letter map: a=0, b=1, ... z=25
def encode_word(word):
    """Encode a word as 5-bit letter tokens + trailing space."""
    tokens = []
    for ch in word.lower():
        if 'a' <= ch <= 'z':
            tokens.append(ord(ch) - 97)  # a=0, b=1, ..., z=25
    if tokens:
        tokens.append(SPACE)  # word separator
    return tokens

def encode_sentence(text):
    """Encode a sentence as word tokens separated by spaces."""
    tokens = []
    for word in text.lower().split():
        tokens.extend(encode_word(word))
    return tokens[:MAX_SEQ]

# Load word list
with open("/workspace/words50k.json") as f:
    WORDS = json.load(f)
print(f"Loaded {len(WORDS)} English words")

# Generate 20K Q/A pairs using real words
qa_pairs = []
for _ in range(20000):
    pattern = random.randint(0, 9)
    w = random.sample(WORDS, 4)
    if pattern == 0: q, a = f"what is {w[0]}", f"it is {w[1]}"
    elif pattern == 1: q, a = f"how to {w[0]}", f"you can {w[0]} by {w[1]}"
    elif pattern == 2: q, a = f"explain {w[0]}", f"{w[0]} means {w[1]}"
    elif pattern == 3: q, a = f"tell me about {w[0]}", f"{w[0]} is {w[1]}"
    elif pattern == 4: q, a = f"what does {w[0]} mean", f"{w[0]} refers to {w[1]}"
    elif pattern == 5: q, a = f"can you {w[0]}", f"yes you can {w[0]} for {w[1]}"
    elif pattern == 6: q, a = f"why is {w[0]} important", f"{w[0]} matters because {w[1]}"
    elif pattern == 7: q, a = f"i like {w[0]}", f"{w[0]} is wonderful"
    elif pattern == 8: q, a = f"what is the best {w[0]}", f"the best {w[0]} is {w[1]}"
    else: q, a = f"how does {w[0]} work", f"{w[0]} works through {w[1]}"
    qa_pairs.append((q, a))

encoded = [(encode_sentence(q), encode_sentence(a)) for q,a in qa_pairs
           if len(encode_sentence(q)) > 3 and len(encode_sentence(a)) > 1]
print(f"Tokenized: {len(encoded)} pairs, max_seq={MAX_SEQ}")

# Model
class WordTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS)
        self.out = nn.Linear(D_MODEL, VOCAB)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.out(self.transformer(x, mask=mask))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WordTransformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
BATCH = 8
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params:,} params, device={device}")

# Train
t0 = time.time()
for ep in range(40):
    total_loss = 0; n = 0; random.shuffle(encoded)
    train_data = encoded[:len(encoded)//2]
    for i in range(0, len(train_data), BATCH):
        batch = train_data[i:i+BATCH]
        if len(batch) < 2: continue
        max_len = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        tgt = torch.full((len(batch), max_len), -1, dtype=torch.long, device=device)
        for j,(q,a) in enumerate(batch):
            c = q[:-1] + a; CL = min(len(c), max_len)
            inp[j,:CL] = torch.tensor(c[:CL], dtype=torch.long)
            tgt[j,:min(len(q[1:]+a),max_len)] = torch.tensor((q[1:]+a)[:min(len(q[1:]+a),max_len)], dtype=torch.long)
        out = model(inp); loss = loss_fn(out.view(-1, VOCAB), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n += 1
    if ep % 10 == 9 or ep < 3:
        acc = sum(1 for q,a in encoded[len(encoded)//2:][:100]
                  if a and model(torch.tensor([q[:MAX_SEQ]], device=device))[0,-1].argmax().item() == a[0])
        print(f"ep{ep+1}: loss{total_loss/max(1,n):.4f} acc{acc}%")

print(f"\nTrained: {time.time()-t0:.1f}s")

# Decode tokens → text
def tokens_to_text(tokens):
    chars = "abcdefghijklmnopqrstuvwxyz"
    result = ""; in_word = False
    for t in tokens:
        if t == 28: break
        if t == SPACE: result += " "; continue
        if 0 <= t <= 25: result += chars[t]
    return result

# Test
print("\n=== CHAT ===")
for q_text in ["what is python","how to program","explain learning","tell me about science"]:
    q = encode_sentence(q_text); gen = list(q)
    with torch.no_grad():
        for _ in range(100):
            if len(gen) >= MAX_SEQ-1: break
            logits = model(torch.tensor([gen], device=device))[0,-1]
            p = logits.argmax().item()
            gen.append(p)
            if p == 28 and len(gen) > len(q)+6: break
    ans = tokens_to_text(gen[len(q):])
    print(f"Q: {q_text}")
    print(f"A: {ans.strip()[:80]}")
    print()
