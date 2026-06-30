#!/usr/bin/env python3
"""5bit Word Vocab Transformer v2 — 60 epochs, fixed chat, natural questions."""
import torch, torch.nn as nn, random, json, time

VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 32, 256, 8, 4, 512
TOKENS_PER_WORD, DATA_TOKENS, SPACE_TOKEN = 3, 28, 31

class WordVocab:
    def __init__(self, words):
        self.w2t = {}; self.t2w = {}
        for i, word in enumerate(words[:DATA_TOKENS**TOKENS_PER_WORD]):
            tokens = []; n = i
            for _ in range(TOKENS_PER_WORD): tokens.append(n % DATA_TOKENS); n //= DATA_TOKENS
            tokens.reverse(); self.w2t[word] = tokens; self.t2w[tuple(tokens)] = word
    def encode(self, text):
        tokens = []
        for word in text.lower().split():
            wt = self.w2t.get(word)
            if wt: tokens.extend(wt + [SPACE_TOKEN])
        return tokens[:MAX_SEQ]
    def decode(self, tokens):
        result = []; buf = []
        for t in tokens:
            if t == SPACE_TOKEN or t >= 28:
                if len(buf) >= TOKENS_PER_WORD:
                    w = self.t2w.get(tuple(buf[:TOKENS_PER_WORD]))
                    if w: result.append(w)
                buf = []
                if t >= 28 and t != SPACE_TOKEN: break
            else: buf.append(t)
        return " ".join(result)

with open("/workspace/words50k.json") as f: WORDS = json.load(f)
vocab = WordVocab(WORDS)
print(f"Vocab: {len(vocab.w2t)} words")

# Generate 25K Q/A pairs
qa_pairs = []
PATTERNS = [
    (["what is {}","how to {}","explain {}","tell me about {}","what does {} mean",
      "can you {}","why is {} important","i like {}","describe {}","how does {} work",
      "what can you tell me about {}","is {} useful","do you know {}","define {}",
      "what is the meaning of {}","how would you explain {}","what should i know about {}"],
     ["it means {}","you can {}","{} is {}","{} means {}","refers to {}",
      "yes {} helps with {}","it matters for {}","{} is great","{} involves {}","it works via {}",
      "{} is about {}","yes {} is useful","{} is known for {}","{} refers to {}",
      "{} means {}","it relates to {}","{} is important for {}"])
]
for _ in range(25000):
    q_tmpl = random.choice(PATTERNS[0][0])
    a_tmpl = random.choice(PATTERNS[0][1])
    n_q = q_tmpl.count("{}")
    n_a = a_tmpl.count("{}")
    w = random.sample(WORDS, max(n_q, n_a+1))
    q = q_tmpl.format(*w[:n_q])
    a = a_tmpl.format(*w[:n_a])
    qa_pairs.append((q, a))

def TL(x): return torch.tensor(x, dtype=torch.long)
encoded = []
for q,a in qa_pairs:
    qe = vocab.encode(q); ae = vocab.encode(a)
    if len(qe) > 2 and len(ae) > 1:
        encoded.append((qe, ae))
print(f"Data: {len(encoded)} pairs")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL); self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS); self.out = nn.Linear(D_MODEL, VOCAB)
    def forward(self, x):
        if x.size(1) == 0: return torch.zeros(1, VOCAB, device=x.device)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.out(self.transformer(self.embed(x)+self.pos(pos),
                mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)))

device = torch.device("cuda"); model = Model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1); BATCH = 10
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

t0 = time.time()
for ep in range(60):
    total_loss = 0; n = 0; random.shuffle(encoded)
    for i in range(0, len(encoded)//2, BATCH):
        batch = encoded[i:i+BATCH]
        if len(batch) < 2: continue
        mx = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.zeros(len(batch), mx, dtype=torch.long, device=device)
        tgt = torch.full((len(batch), mx), -1, dtype=torch.long, device=device)
        for j,(q,a) in enumerate(batch):
            c = q[:-1]+a; CL=min(len(c),mx)
            inp[j,:CL] = TL(c[:CL]); tgt[j,:CL] = TL(c[1:]+[a[-1]])[:CL]
        out = model(inp); loss = loss_fn(out.view(-1,VOCAB), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n += 1
    if ep%15==14 or ep<3:
        correct = 0; total = 0
        for q,a in encoded[len(encoded)//2:][:200]:
            if len(q)==0: continue
            pred = model(TL(q).unsqueeze(0).to(device))[0,-1].argmax().item()
            if a and pred == a[0]: correct += 1
            total += 1
        acc = correct/max(1,total)
        print(f"ep{ep+1}: loss{total_loss/max(1,n):.4f} acc{acc:.1%}")

print(f"Trained: {time.time()-t0:.1f}s")

# Chat
def chat(model, text):
    tokens = vocab.encode(text)
    if not tokens: return "..."
    gen = list(tokens)
    with torch.no_grad():
        control_count = 0
        for _ in range(120):
            if len(gen) >= MAX_SEQ-20: break
            inp = TL(gen).unsqueeze(0).to(device)
            if inp.size(1) == 0: break
            logits = model(inp)[0,-1]
            p = logits.argmax().item()
            gen.append(p)
            if p >= 28: control_count += 1
            else: control_count = 0
            if control_count >= 3 and len(gen) > len(tokens)+3: break
    return vocab.decode(gen[len(tokens):])

print("\n=== CHAT ===")
for q_text in [
    "what is python",
    "how to program",
    "explain coding",
    "can python help me to program",
    "is python useful",
    "tell me about computers"
]:
    ans = chat(model, q_text)
    print(f"Q: {q_text}")
    print(f"A: {ans}")
    print()
