#!/usr/bin/env python3
"""5bit Transformer Chat — grammar tags + repetition penalty decoder."""
import torch, torch.nn as nn, random, json, time

VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 32, 256, 8, 4, 128
WORD_MAP = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ ."): WORD_MAP[c] = i
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz@-"): WORD_MAP[c] = i

def encode(text):
    tokens = []
    for word in text.split():
        tokens.append(26); tokens.append(31)
        for ch in word:
            if ch in WORD_MAP: tokens.append(WORD_MAP[ch])
        tokens.append(30)
    return tokens[:MAX_SEQ]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS)
        self.out = nn.Linear(D_MODEL, VOCAB)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.out(self.transformer(self.embed(x)+self.pos(pos),
                mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)))

def decode_with_penalty(model, q, max_new=60, penalty=10.0):
    gen = list(q); seen = {}
    with torch.no_grad():
        for _ in range(max_new):
            if len(gen) >= MAX_SEQ-1: break
            logits = model(torch.tensor([gen], device=device))[0,-1]
            for tok, count in seen.items():
                if count > 2: logits[tok] -= penalty * (count - 1)
            p = logits.argmax().item()
            gen.append(p)
            seen[p] = seen.get(p, 0) + 1
            if p == 28 and len(gen) > len(q) + 6: break
    return gen

def tokens_to_text(tokens):
    rev = {v:k for k,v in WORD_MAP.items()}
    words = []; cur = ""; in_word = False
    for t in tokens:
        if t == 28: break
        if t in (26,27):
            if cur: words.append(cur); cur = ""
            in_word = False; continue
        if t == 31: in_word = True; continue
        if t == 30:
            if cur: words.append(cur); cur = ""
            in_word = False; continue
        if in_word: cur += rev.get(t, "?")
    if cur: words.append(cur)
    return " ".join(words)

device = "cuda"
with open("/workspace/dialog.json") as f: qa_pairs = json.load(f)
encoded = [(encode(q), encode(a)) for q,a in qa_pairs[:8000] if len(encode(q))>3 and len(encode(a))>1]
print(f"Training on {len(encoded)} pairs")

model = Model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
BATCH = 16

t0 = time.time()
for ep in range(25):
    random.shuffle(encoded)
    for i in range(0, len(encoded)//2, BATCH):
        batch = encoded[i:i+BATCH]
        max_len = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.zeros(len(batch),max_len,dtype=torch.long,device=device)
        tgt = torch.full((len(batch),max_len),-1,dtype=torch.long,device=device)
        for j,(q,a) in enumerate(batch):
            c = q[:-1]+a; CL=min(len(c),max_len); inp[j,:CL]=torch.tensor(c[:CL])
            tgt[j,:min(len(q[1:]+a),max_len)]=torch.tensor((q[1:]+a)[:min(len(q[1:]+a),max_len)])
        out = model(inp); loss = loss_fn(out.view(-1,VOCAB), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()

print(f"Trained: {time.time()-t0:.1f}s")

print("\n=== CHAT ===")
for q_text in ["what is python","how do i use functions","explain decorator",
               "difference between list and tuple","give example of lambda",
               "is flask important"]:
    q = encode(q_text)
    gen = decode_with_penalty(model, q)
    ans = tokens_to_text(gen[len(q):])
    print(f"Q: {q_text}")
    print(f"A: {ans.strip() or '(empty)'}")
    print()
