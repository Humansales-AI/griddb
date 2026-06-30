#!/usr/bin/env python3
"""5bit Nested Transformer — 8K word vocab, word-level IDs, letter LUT."""
import torch, torch.nn as nn, random, json, time

VOCAB_SIZE = 8192
D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 256, 8, 4, 512

with open("/workspace/words_freq.json") as f: WORDS = json.load(f)
lut_fwd = {}; lut_rev = {}
for i, word in enumerate(WORDS[:VOCAB_SIZE]):
    lut_fwd[word] = i; lut_rev[i] = word
UNK = VOCAB_SIZE

def encode(text):
    return [lut_fwd.get(w, UNK) for w in text.lower().split()][:MAX_SEQ]
def decode(ids):
    return " ".join(lut_rev.get(i, "?") for i in ids if i < VOCAB_SIZE)

with open("/workspace/real_dialog_v2.json") as f: qa_pairs = json.load(f)

def TL(x): return torch.tensor(x, dtype=torch.long, device="cuda")
encoded = [(encode(q), encode(a)) for q,a in qa_pairs if encode(q) and encode(a)]
print(f"Vocab: {VOCAB_SIZE}, Dialog: {len(encoded)} pairs")
print(f"First: {list(lut_rev.values())[:10]}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE+1, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS)
        self.out = nn.Linear(D_MODEL, VOCAB_SIZE+1)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.out(self.transformer(self.embed(x)+self.pos(pos),
            mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)))

model = Model().to("cuda")
opt = torch.optim.AdamW(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

t0 = time.time()
for ep in range(60):
    total_loss = 0; n = 0; random.shuffle(encoded)
    for i in range(0, len(encoded)//2, 12):
        batch = encoded[i:i+12]
        if len(batch)<2: continue
        mx = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.full((len(batch),mx), UNK, dtype=torch.long, device="cuda")
        tgt = torch.full((len(batch),mx), -1, dtype=torch.long, device="cuda")
        for j,(q,a) in enumerate(batch):
            c = q[:-1]+a; CL=min(len(c),mx)
            inp[j,:CL]=TL(c[:CL]); tgt[j,:CL]=TL(c[1:]+[a[-1]])[:CL]
        out = model(inp); loss = loss_fn(out.view(-1, VOCAB_SIZE+1), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n += 1
    if ep%20==19 or ep<3:
        correct=0; total=0
        for q,a in encoded[len(encoded)//2:][:200]:
            if len(q)==0: continue
            p = model(TL(q).unsqueeze(0))[0,-1].argmax().item()
            if a and p==a[0]: correct+=1; total+=1
        print(f"ep{ep+1}: loss{total_loss/max(1,n):.4f} acc{correct/max(1,total):.1%}")

print(f"Trained: {time.time()-t0:.0f}s")

def respond(text):
    ids = encode(text)
    if not ids: return "(oov)"
    gen = list(ids)
    with torch.no_grad():
        cc = 0
        for _ in range(80):
            if len(gen)>=MAX_SEQ-5: break
            logits = model(TL(gen).unsqueeze(0))[0,-1]
            probs = torch.softmax(logits/0.7, dim=-1)
            top10 = torch.topk(probs, 10).indices
            p = top10[random.randint(0,min(5,len(top10)-1))].item()
            gen.append(p)
            if p>=VOCAB_SIZE: cc+=1
            else: cc=0
            if cc>=2: break
    return decode(gen[len(ids):])

print()
for q_text in ["hello how are you","what do you like","i love this place","tell me something","that sounds good"]:
    print(f"Q: {q_text}")
    print(f"A: {respond(q_text)}")
    print()
