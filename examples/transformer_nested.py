#!/usr/bin/env python3
"""5bit Nested Transformer — Word IDs from LUT, transformer sees word-level tokens."""
import torch, torch.nn as nn, random, json, time

VOCAB_SIZE = 32        # 5-bit tokens
D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 256, 8, 4, 512
SPACE_TOKEN = 26       # word boundary in 5-bit

# ── LUT: word → letter tokens → word ID ──────────────────────────
class NestedLUT:
    def __init__(self, words):
        self.id_to_letters = {}   # word_id → [c, a, t]
        self.letters_to_id = {}   # (c, a, t) → word_id
        self.word_to_id = {}      # "cat" → 42
        self.id_to_word = {}      # 42 → "cat"

        for i, word in enumerate(words):
            tokens = tuple(ord(ch)-97 for ch in word.lower() if 'a'<=ch<='z')
            if 1 <= len(tokens) <= 12 and tokens not in self.letters_to_id:
                self.id_to_letters[i] = list(tokens)
                self.letters_to_id[tokens] = i
                self.word_to_id[word] = i
                self.id_to_word[i] = word

    def encode(self, text):
        """Sentence → word ID sequence with space separators."""
        tokens = []
        for word in text.lower().split():
            wid = self.word_to_id.get(word)
            if wid is not None: tokens.append(wid)
        return tokens[:MAX_SEQ]  # word IDs only, no letter tokens

    def decode(self, ids):
        """Word ID sequence → sentence."""
        return ' '.join(self.id_to_word.get(wid, '?') for wid in ids)

    def __len__(self): return len(self.word_to_id)

# ── Load ─────────────────────────────────────────────────────────
with open("/workspace/words_freq.json") as f: WORDS = json.load(f)
lut = NestedLUT(WORDS)
print(f"LUT: {len(lut)} words | cat={lut.word_to_id.get('cat','?')} hello={lut.word_to_id.get('hello','?')}")

with open("/workspace/real_dialog_v2.json") as f: qa_pairs = json.load(f)

def TL(x): return torch.tensor(x, dtype=torch.long)
encoded = [(lut.encode(q), lut.encode(a)) for q,a in qa_pairs if lut.encode(q) and lut.encode(a)]
print(f"Dialog: {len(encoded)} pairs, max word ID: {max(max(q) for q,_ in encoded)}")

# ── Model (word-level, like GPT) ─────────────────────────────────
class NestedTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS)
        self.out = nn.Linear(D_MODEL, vocab_size)
    def forward(self, x):
        if x.size(1)==0: return torch.zeros(1, vocab_size, device=x.device)
        pos = torch.arange(x.size(1),device=x.device).unsqueeze(0)
        return self.out(self.transformer(self.embed(x)+self.pos(pos),
            mask=nn.Transformer.generate_square_subsequent_mask(x.size(1),device=x.device)))

vocab_size = len(lut)  # ~50K word vocabulary
model = NestedTransformer(vocab_size).to("cuda")
opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
print(f"Params: {sum(p.numel() for p in model.parameters()):,} | Vocab: {vocab_size} words")

# ── Train ────────────────────────────────────────────────────────
t0 = time.time()
for ep in range(60):
    total_loss = 0; n = 0; random.shuffle(encoded)
    for i in range(0, len(encoded)//2, 8):
        batch = encoded[i:i+8]
        if len(batch)<2: continue
        mx = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.full((len(batch),mx), vocab_size-1, dtype=torch.long, device="cuda")
        tgt = torch.full((len(batch),mx), -1, dtype=torch.long, device="cuda")
        for j,(q,a) in enumerate(batch):
            c = q[:-1] + a; CL = min(len(c), mx)
            inp[j,:CL] = TL(c[:CL])
            tgt[j,:CL] = TL(c[1:] + [a[-1]])[:CL]
        out = model(inp); loss = loss_fn(out.view(-1, vocab_size), tgt.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n += 1
    if ep%20==19 or ep<3:
        correct=0; total=0
        for q,a in encoded[len(encoded)//2:][:200]:
            if len(q)==0: continue
            p = model(TL(q).unsqueeze(0).to("cuda"))[0,-1].argmax().item()
            if a and p==a[0]: correct+=1; total+=1
        print(f"ep{ep+1}: loss{total_loss/max(1,n):.4f} acc{correct/max(1,total):.1%}")

print(f"Trained: {time.time()-t0:.0f}s")

# ── Chat ─────────────────────────────────────────────────────────
def respond(text):
    ids = lut.encode(text)
    if not ids: return "(oov)"
    gen = list(ids)
    with torch.no_grad():
        cc = 0
        for _ in range(80):
            if len(gen)>=MAX_SEQ-5: break
            inp = TL(gen).unsqueeze(0).to("cuda")
            if inp.size(1)==0: break
            logits = model(inp)[0,-1]
            probs = torch.softmax(logits/0.7, dim=-1)
            top10 = torch.topk(probs, 10).indices
            p = top10[random.randint(0,min(5,len(top10)-1))].item()
            gen.append(p)
            if p >= vocab_size-1: cc+=1
            else: cc=0
            if cc>=2: break
    return lut.decode(gen[len(ids):])

print()
for q_text in ["hello how are you","what do you like","i love this place","tell me something","that sounds good","where are you from","do you like it here"]:
    ans = respond(q_text)
    print(f"Q: {q_text}")
    print(f"A: {ans}")
    print()
