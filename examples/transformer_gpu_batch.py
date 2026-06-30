import torch, torch.nn as nn, random, time

VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 32, 256, 8, 4, 128
BATCH = 8

WORD_MAP = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ ."): WORD_MAP[c] = i
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz@-"): WORD_MAP[c] = i

def encode_5bit(text):
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

qa_pairs = [("capital of france","paris"),("days in a week","seven"),("color of the sky","blue"),
    ("author of hamlet","shakespeare"),("largest ocean","pacific"),("hottest planet","venus"),
    ("closest star","sun"),("largest animal","whale"),("fastest animal","cheetah"),
    ("smallest bird","hummingbird")]*50
encoded = [(encode_5bit(q), encode_5bit(a)) for q,a in qa_pairs if encode_5bit(q) and encode_5bit(a)]

class FiveBitTransformer(nn.Module):
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

device = torch.device("cuda")
model = FiveBitTransformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
print(f"{len(encoded)} pairs, batch={BATCH}, {sum(p.numel() for p in model.parameters()):,} params")

t0 = time.time()
for ep in range(80):
    total_loss = 0; n = 0; random.shuffle(encoded)
    for i in range(0, len(encoded)//2, BATCH):
        batch = encoded[i:i+BATCH]
        max_len = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        tgt = torch.full((len(batch), max_len), -1, dtype=torch.long, device=device)
        for j,(q,a) in enumerate(batch):
            combined = q[:-1] + a; CL = min(len(combined), max_len)
            inp[j,:CL] = torch.tensor(combined[:CL])
            qa = q[1:] + a; tgt[j,:min(len(qa),max_len)] = torch.tensor(qa[:min(len(qa),max_len)])
        out = model(inp)
        loss = loss_fn(out.view(-1, VOCAB), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n += 1
    if ep % 20 == 19 or ep == 0:
        correct = sum(1 for q,a in encoded[len(encoded)//2:] if a and model(torch.tensor([q[:MAX_SEQ]],device=device))[0,-1].argmax().item()==a[0])
        print(f"ep {ep+1}: loss {total_loss/max(1,n):.4f}  acc {correct/max(1,len(encoded)//2):.1%}")

print(f"Train: {time.time()-t0:.1f}s")
rev = {v:k for k,v in WORD_MAP.items()}
for q_text in ["capital of france","days in a week","color of the sky"]:
    q = encode_5bit(q_text); gen = list(q)
    with torch.no_grad():
        for _ in range(30):
            if len(gen)>=MAX_SEQ-1: break
            p = model(torch.tensor([gen],device=device))[0,-1].argmax().item()
            gen.append(p); 
            if p>=28: break
    ans = "".join(rev.get(t,"?") for t in gen[len(q):] if t<28)
    print(f"Q: {q_text} -> {ans.strip()}")
