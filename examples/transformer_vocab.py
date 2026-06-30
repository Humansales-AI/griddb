#!/usr/bin/env python3
"""5bit Word-Level Transformer — 50K word vocabulary, 28^3 token encoding."""
import torch, torch.nn as nn, random, json, time

VOCAB = 32; D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 256, 8, 4, 256
TOKENS_PER_WORD = 3; DATA_TOKENS = 28; SPACE_TOKEN = 31

class WordVocab:
    def __init__(self, words):
        self.w2id = {}; self.id2w = {}; self.w2t = {}; self.t2w = {}
        for i, word in enumerate(words[:DATA_TOKENS**TOKENS_PER_WORD]):
            self.w2id[word] = i; self.id2w[i] = word
            tokens = []; n = i
            for _ in range(TOKENS_PER_WORD): tokens.append(n % DATA_TOKENS); n //= DATA_TOKENS
            tokens.reverse(); self.w2t[word] = tokens; self.t2w[tuple(tokens)] = word
    def encode(self, text):
        tokens = []
        for word in text.lower().split():
            wt = self.w2t.get(word)
            if wt: tokens.extend(wt); tokens.append(SPACE_TOKEN)
        return tokens[:MAX_SEQ]
    def decode(self, tokens):
        result = []; buf = []
        for t in tokens:
            if t == SPACE_TOKEN or t >= 28:
                if len(buf) >= TOKENS_PER_WORD:
                    w = self.t2w.get(tuple(buf[:TOKENS_PER_WORD]))
                    if w: result.append(w)
                buf = [];
                if t >= 28 and t != SPACE_TOKEN: break
            else: buf.append(t)
        return " ".join(result)
    def __len__(self): return len(self.w2id)

with open("/workspace/words50k.json") as f: WORDS = json.load(f)
vocab = WordVocab(WORDS)
print(f"Vocab: {len(vocab)} words ({TOKENS_PER_WORD} tokens/word, {DATA_TOKENS}^{TOKENS_PER_WORD}={DATA_TOKENS**TOKENS_PER_WORD})")

# Generate training data
qa_pairs = []
for _ in range(15000):
    p = random.randint(0, 9); w = random.sample(WORDS, 4)
    if p == 0: q, a = f"what is {w[0]}", f"it means {w[1]}"
    elif p == 1: q, a = f"how to {w[0]}", f"you can {w[1]}"
    elif p == 2: q, a = f"explain {w[0]}", f"{w[0]} is {w[1]}"
    elif p == 3: q, a = f"tell about {w[0]}", f"{w[0]} means {w[1]}"
    elif p == 4: q, a = f"what is {w[0]} mean", f"refers to {w[1]}"
    elif p == 5: q, a = f"can you {w[0]}", f"yes by {w[1]}"
    elif p == 6: q, a = f"why {w[0]} important", f"because of {w[1]}"
    elif p == 7: q, a = f"i like {w[0]}", f"{w[0]} is great"
    elif p == 8: q, a = f"best {w[0]}", f"the best is {w[1]}"
    else: q, a = f"how does {w[0]} work", f"it works via {w[1]}"
    qa_pairs.append((q, a))

def to_long(x): return torch.tensor(x, dtype=torch.long)

encoded = [(vocab.encode(q), vocab.encode(a)) for q,a in qa_pairs
           if len(vocab.encode(q)) > 2 and len(vocab.encode(a)) > 1]
print(f"Data: {len(encoded)} pairs")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL); self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS); self.out = nn.Linear(D_MODEL, VOCAB)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.out(self.transformer(self.embed(x)+self.pos(pos),
                mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)))

device = torch.device("cuda"); model = Model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1); BATCH = 8
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

t0 = time.time()
for ep in range(30):
    total_loss = 0; n = 0; random.shuffle(encoded)
    for i in range(0, len(encoded)//2, BATCH):
        batch = encoded[i:i+BATCH]
        if len(batch) < 2: continue
        mx = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.zeros(len(batch), mx, dtype=torch.long, device=device)
        tgt = torch.full((len(batch), mx), -1, dtype=torch.long, device=device)
        for j,(q,a) in enumerate(batch):
            c = q[:-1]+a; CL=min(len(c),mx)
            inp[j,:CL] = to_long(c[:CL]).to(device)
            tgt_full = q[1:]+a; tgt[j,:min(len(tgt_full),mx)] = to_long(tgt_full[:min(len(tgt_full),mx)])
        out = model(inp); loss = loss_fn(out.view(-1,VOCAB), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n += 1
    if ep%10==9 or ep<3:
        acc = sum(1 for q,a in encoded[len(encoded)//2:][:100]
                  if a and model(to_long(q[:MAX_SEQ]).unsqueeze(0).to(device))[0,-1].argmax().item() == a[0])
        print(f"ep{ep+1}: loss{total_loss/max(1,n):.4f} acc{acc}%")

print(f"Train: {time.time()-t0:.1f}s")

print("\n=== CHAT ===")
for q_text in ["what is python","how to program","explain learning","tell about science"]:
    q = vocab.encode(q_text); gen = list(q)
    with torch.no_grad():
        for _ in range(60):
            if len(gen) >= MAX_SEQ-1: break
            logits = model(to_long(gen).unsqueeze(0).to(device))[0,-1]
            p = logits.argmax().item(); gen.append(p)
            if p >= 28 and len(gen) > len(q)+3: break
    ans = vocab.decode(gen[len(q):])
    raw = model(to_long(q).unsqueeze(0).to(device))[0,-1].argmax().item()
    print(f"Q: {q_text}")
    print(f"  pred={raw} A: {ans}")
    print()
