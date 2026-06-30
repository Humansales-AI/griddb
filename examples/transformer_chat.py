#!/usr/bin/env python3
"""5bit Transformer Chat — 32-token vocab, grammar tags, top-k sampling."""
import torch, torch.nn as nn, random, json, time

VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 32, 256, 8, 4, 128
BATCH = 20

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

class FiveBitChat(nn.Module):
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

def decode(tokens):
    rev = {v: k for k, v in WORD_MAP.items()}
    words = []; cur = ""; in_word = False
    for t in tokens:
        if t == 28: break
        if t in (26, 27):
            if cur: words.append(cur); cur = ""
            in_word = False; continue
        if t == 31: in_word = True; continue
        if t == 30:
            if cur: words.append(cur); cur = ""
            in_word = False; continue
        if in_word: cur += rev.get(t, "?")
    if cur: words.append(cur)
    return " ".join(words)

def main():
    with open("/workspace/dialog.json") as f:
        qa_pairs = json.load(f)
    qa_pairs = qa_pairs[:8000]
    encoded = [(encode(q), encode(a)) for q, a in qa_pairs if len(encode(q)) > 3 and len(encode(a)) > 1]
    print(f"Training on {len(encoded)} pairs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FiveBitChat().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    t0 = time.time()
    for ep in range(25):
        total_loss = 0; n = 0; random.shuffle(encoded)
        for i in range(0, len(encoded) // 2, BATCH):
            batch = encoded[i:i+BATCH]
            max_len = min(max(len(q)+len(a) for q, a in batch), MAX_SEQ)
            inp = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            tgt = torch.full((len(batch), max_len), -1, dtype=torch.long, device=device)
            for j, (q, a) in enumerate(batch):
                c = q[:-1] + a; CL = min(len(c), max_len)
                inp[j, :CL] = torch.tensor(c[:CL])
                tgt[j, :min(len(q[1:]+a), max_len)] = torch.tensor((q[1:]+a)[:min(len(q[1:]+a), max_len)])
            out = model(inp); loss = loss_fn(out.view(-1, VOCAB), tgt.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
    acc = sum(1 for q, a in encoded[len(encoded)//2:][:100]
              if a and model(torch.tensor([q[:MAX_SEQ]], device=device))[0, -1].argmax().item() == a[0])
    print(f"Trained: {time.time()-t0:.1f}s  acc: {acc}%")

    # Chat test
    tests = ["what is python", "how do i use functions", "explain decorator",
             "difference between list and tuple", "give example of lambda",
             "why learn asyncio", "is flask important", "when to use numpy"]
    rev = {v: k for k, v in WORD_MAP.items()}
    print("\n=== CHAT ===")
    for q_text in tests:
        q = encode(q_text); gen = list(q)
        with torch.no_grad():
            for _ in range(150):
                if len(gen) >= MAX_SEQ - 1: break
                logits = model(torch.tensor([gen], device=device))[0, -1]
                top5 = torch.topk(logits, 5).indices
                p = top5[random.randint(0, min(3, len(top5)-1))].item()
                gen.append(p)
                if p == 28 and len(gen) > len(q) + 6: break
        raw = model(torch.tensor([q], device=device))[0, -1].argmax().item()
        ans = decode(gen[len(q):])
        print(f"Q: {q_text}")
        print(f"  pred={raw} A: {ans if ans else '(empty)'}")
    print()

if __name__ == "__main__":
    main()
