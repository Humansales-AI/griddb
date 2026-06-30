#!/usr/bin/env python3
"""
5bit Transformer with Grammar Layer — 10K Q/A pairs
=====================================================
Grammar tags prepended to each word before attention.
[N] = noun, [V] = verb, [O] = other.
Run on GPU: python3 transformer_grammar.py
"""
import torch, torch.nn as nn, random, time

VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 32, 256, 8, 4, 128
BATCH = 16

WORD_MAP = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ ."): WORD_MAP[c] = i
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz@-"): WORD_MAP[c] = i

def encode_5bit_grammar(text):
    """Encode with grammar tags: 26=[N], 27=[V] prepended before each word."""
    tokens = []
    words = text.split()
    for wi, word in enumerate(words):
        if wi < 2: tokens.append(26)  # First words = noun/subject
        elif wi == len(words)-1 and len(words) <= 4: tokens.append(27)  # Last verb
        else: tokens.append(26)

        tokens.append(31); in_special = False
        for ch in word:
            if ch in WORD_MAP:
                if in_special: tokens.append(30); in_special = False
                tokens.append(WORD_MAP[ch])
            elif ch.isdigit():
                if in_special: tokens.append(30); in_special = False
                tokens.extend([30, int(ch), 31])
        if in_special: tokens.append(30)
        tokens.append(30)
    return tokens[:MAX_SEQ]

WORDS = [w.strip() for w in """what is the capital of france paris how many days in a week seven color sky blue who wrote hamlet shakespeare water made hydrogen oxygen continents are there largest ocean pacific painted mona lisa davinci speed light three hundred thousand kilometers per second planets solar system eight animal whale tallest mountain everest hottest planet venus closest star sun first president washington author romeo juliet element gold currency japan yen europe euro britain pound language brazil portuguese germany german china mandarin india hindi coldest continent antarctica deepest mariana trench longest river nile country canada russia australia mexico spain italy smallest bird hummingbird fastest cheetah atom proton neutron electron dna gene cell virus bacteria red green yellow black white heavy light soft hard tall short big small young old new fast slow hot cold wet dry rich poor happy sad brave afraid wise foolish kind cruel clean dirty loud quiet strong weak thick thin full empty open closed near far early late fresh stale sweet sour round flat straight curved alive dead safe dangerous simple complex gentle rough polite rude lucky unlucky calm angry""".split()]

TEMPLATES = [
    ("what is the {0} of {1}", 3), ("how many {0} in a {1}", 3),
    ("what {0} is the {1}", 3), ("what is the {0} planet", 2),
    ("what is the {0} animal", 2), ("what is the {0} continent", 2),
    ("what is the {0} country", 2), ("how many {0} are there", 2),
    ("who {0} {1}", 3), ("what is {0} made of", 2),
    ("what is the {0} element", 2), ("what currency does {0} use", 2),
    ("what language does {0} speak", 2), ("is {0} hot or cold", 2),
    ("is {0} big or small", 2),
]

qa_pairs = []
for _ in range(10000):
    tmpl, n = random.choice(TEMPLATES)
    words = random.sample(WORDS, n)
    question = tmpl.format(*words[:-1])
    qa_pairs.append((question, words[-1]))
print(f"Generated {len(qa_pairs)} Q/A pairs")

encoded = [(encode_5bit_grammar(q), encode_5bit_grammar(a)) for q,a in qa_pairs]
encoded = [(q,a) for q,a in encoded if len(q)>5 and len(a)>2]
print(f"Encoded: {len(encoded)} grammar-tagged pairs, batch={BATCH}")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FiveBitTransformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params:,} params, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

t0 = time.time()
best_acc = 0
for ep in range(100):
    total_loss = 0; n = 0; random.shuffle(encoded)
    for i in range(0, len(encoded)//2, BATCH):
        batch = encoded[i:i+BATCH]
        if len(batch) < 2: continue
        max_len = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
        inp = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        tgt = torch.full((len(batch), max_len), -1, dtype=torch.long, device=device)
        for j,(q,a) in enumerate(batch):
            combined = q[:-1] + a; CL = min(len(combined), max_len)
            inp[j,:CL] = torch.tensor(combined[:CL])
            qa_tgt = q[1:] + a; tgt[j,:min(len(qa_tgt),max_len)] = torch.tensor(qa_tgt[:min(len(qa_tgt),max_len)])
        out = model(inp)
        loss = loss_fn(out.view(-1, VOCAB), tgt.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); n += 1
    scheduler.step()
    if ep % 25 == 24 or ep == 0:
        correct = 0; total = 0
        for q, a in encoded[len(encoded)//2:len(encoded)//2+200]:
            CL = min(len(q), MAX_SEQ)
            inp = torch.tensor([q[:CL]], device=device)
            with torch.no_grad():
                pred = model(inp)[0,-1].argmax().item()
            if a and pred < 28 and a[0] < 28 and pred == a[0]: correct += 1
            total += 1
        acc = correct/max(1,total)
        if acc > best_acc: best_acc = acc
        print(f"ep {ep+1}: loss {total_loss/max(1,n):.4f}  acc {acc:.1%}  best {best_acc:.1%}")

tt = time.time()-t0
print(f"\nTraining: {tt:.1f}s ({tt/60:.1f}min)  Best acc: {best_acc:.1%}")

rev = {v:k for k,v in WORD_MAP.items()}
print("\n" + "="*50)
for q_text in ["capital of france","days in a week","color of the sky","largest ocean","hottest planet","closest star"]:
    q = encode_5bit_grammar(q_text); gen = list(q)
    with torch.no_grad():
        for _ in range(60):
            if len(gen)>=MAX_SEQ-1: break
            p = model(torch.tensor([gen],device=device))[0,-1].argmax().item()
            gen.append(p)
            if p>=28 and len(gen)>len(q)+5: break
    ans_tokens = gen[len(q):]
    ans = ""
    for t in ans_tokens:
        if t >= 28: break
        if t == 26: ans += " [N] "; continue
        if t == 27: ans += " [V] "; continue
        if t == 31 or t == 30: continue
        ans += rev.get(t, "?")
    print(f"Q: {q_text}")
    print(f"A: {ans.strip()}")
    print()
