#!/usr/bin/env python3
"""5bit LUT (Lookup Table) Transformer — letter-based encoding, 50K word LUT."""
import torch, torch.nn as nn, random, json, time

VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 32, 256, 8, 4, 512
SPACE = 26

def word_to_tokens(word):
    return [ord(ch)-97 for ch in word.lower() if 'a' <= ch <= 'z']

with open("/workspace/words_freq.json") as f: WORDS = json.load(f)
WORD_TO_TOKENS = {}; TOKENS_TO_WORD = {}
for word in WORDS[:50000]:
    tokens = word_to_tokens(word)
    if 1 <= len(tokens) <= 12:
        WORD_TO_TOKENS[word] = tokens
        TOKENS_TO_WORD[tuple(tokens)] = word

def encode(text):
    tokens = []
    for word in text.lower().split():
        wt = WORD_TO_TOKENS.get(word)
        if wt: tokens.extend(wt); tokens.append(SPACE)
    return tokens[:MAX_SEQ]

def decode(tokens):
    result = []; buf = []
    for t in tokens:
        if t == SPACE or t >= 28:
            w = TOKENS_TO_WORD.get(tuple(buf))
            if w: result.append(w)
            buf = []
            if t >= 28: break
        else: buf.append(t)
    w = TOKENS_TO_WORD.get(tuple(buf))
    if w: result.append(w)
    return " ".join(result)

print(f"LUT: {len(WORD_TO_TOKENS)} words")
print(f"cat={word_to_tokens('cat')}, hello={word_to_tokens('hello')}")

with open("/workspace/real_dialog_v2.json") as f: qa_pairs = json.load(f)

def TL(x): return torch.tensor(x, dtype=torch.long)
encoded = [(encode(q), encode(a)) for q,a in qa_pairs if len(encode(q))>2 and len(encode(a))>1]
print(f"Dialog: {len(encoded)} pairs")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL); self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS); self.out = nn.Linear(D_MODEL, VOCAB)
    def forward(self, x):
        if x.size(1)==0: return torch.zeros(1,VOCAB,device=x.device)
        pos = torch.arange(x.size(1),device=x.device).unsqueeze(0)
        return self.out(self.transformer(self.embed(x)+self.pos(pos),
            mask=nn.Transformer.generate_square_subsequent_mask(x.size(1),device=x.device)))

device = torch.device("cuda"); model = Model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

t0 = time.time()
for ep in range(80):
    total_loss = 0; n = 0; random.shuffle(encoded)
    for i in range(0, len(encoded)//2, 10):
        batch = encoded[i:i+10]
        if len(batch)<2: continue
        mx = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.zeros(len(batch),mx,dtype=torch.long,device=device)
        tgt = torch.full((len(batch),mx),-1,dtype=torch.long,device=device)
        for j,(q,a) in enumerate(batch):
            c = q[:-1]+a; CL=min(len(c),mx); inp[j,:CL]=TL(c[:CL]); tgt[j,:CL]=TL(c[1:]+[a[-1]])[:CL]
        out = model(inp); loss = loss_fn(out.view(-1,VOCAB), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n += 1
    if ep%20==19 or ep<3:
        correct=0; total=0
        for q,a in encoded[len(encoded)//2:][:200]:
            if len(q)==0: continue
            p = model(TL(q).unsqueeze(0).to(device))[0,-1].argmax().item()
            if a and p==a[0]: correct+=1; total+=1
        print(f"ep{ep+1}: loss{total_loss/max(1,n):.4f} acc{correct/max(1,total):.1%}")

print(f"Trained: {time.time()-t0:.0f}s")

def respond(text):
    tokens = encode(text)
    if not tokens: return "(oov)"
    gen = list(tokens)
    with torch.no_grad():
        cc = 0
        for _ in range(250):
            if len(gen)>=MAX_SEQ-20: break
            inp = TL(gen).unsqueeze(0).to(device)
            if inp.size(1)==0: break
            logits = model(inp)[0,-1]
            probs = torch.softmax(logits / 0.7, dim=-1)
            top5 = torch.topk(probs, 5).indices
            p = top5[random.randint(0, min(4, len(top5)-1))].item()
            gen.append(p)
            if p>=28: cc+=1
            else: cc=0
            if cc>=3 and len(gen)>len(tokens)+5: break
    return decode(gen[len(tokens):])

print()
tests = ["hello how are you","what do you like","i love this place",
         "tell me something","that sounds good","where are you from",
         "i feel great today","do you like it here"]
for q_text in tests:
    ans = respond(q_text)
    print(f"Q: {q_text}")
    print(f"A: {ans}")
    print()
