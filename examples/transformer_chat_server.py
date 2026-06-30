#!/usr/bin/env python3
"""5bit Chat Server — trains 50K word model then serves HTTP chat endpoint."""
import torch, torch.nn as nn, random, json, time, os
from http.server import HTTPServer, BaseHTTPRequestHandler

VOCAB_SIZE = 50001
D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ = 256, 8, 4, 512
MODEL_PATH = "/workspace/chat_model.pt"
PORT = 8080

with open("/workspace/words_freq.json") as f: WORDS = json.load(f)
lut_fwd = {}; lut_rev = {}
for i, word in enumerate(WORDS[:50000]):
    lut_fwd[word] = i; lut_rev[i] = word

def encode(text):
    return [lut_fwd.get(w, 50000) for w in text.lower().split()][:MAX_SEQ]
def decode(ids):
    return " ".join(lut_rev.get(i, "?") for i in ids if i < 50000)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL); self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 1024, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS)
        self.out = nn.Linear(D_MODEL, VOCAB_SIZE)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.out(self.transformer(self.embed(x)+self.pos(pos),
            mask=nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)))

device = torch.device("cuda")
model = Model().to(device)

# Train or load
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded saved model ({MODEL_PATH})")
else:
    with open("/workspace/real_dialog_v2.json") as f: qa_pairs = json.load(f)
    encoded = [(encode(q), encode(a)) for q,a in qa_pairs if encode(q) and encode(a)]
    print(f"Training on {len(encoded)} pairs, {VOCAB_SIZE} vocab")

    opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    for ep in range(60):
        total_loss = 0; n = 0; random.shuffle(encoded)
        for i in range(0, len(encoded)//2, 8):
            batch = encoded[i:i+8]
            if len(batch)<2: continue
            mx = min(max(len(q)+len(a) for q,a in batch), MAX_SEQ)
            inp = torch.full((len(batch),mx), 50000, dtype=torch.long, device=device)
            tgt = torch.full((len(batch),mx), -1, dtype=torch.long, device=device)
            for j,(q,a) in enumerate(batch):
                c = q[:-1]+a; CL=min(len(c),mx)
                inp[j,:CL] = torch.tensor(c[:CL], dtype=torch.long, device=device)
                tgt[j,:CL] = torch.tensor(c[1:]+[a[-1]], dtype=torch.long, device=device)[:CL]
            out = model(inp); loss = loss_fn(out.view(-1, VOCAB_SIZE), tgt.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
        if ep%15==14 or ep<3:
            print(f"ep{ep+1}: loss{total_loss/max(1,n):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

# Chat function
def respond(text, temp=0.7, max_new=100):
    ids = encode(text)
    if not ids: return "(no words in vocabulary)"
    gen = list(ids)
    with torch.no_grad():
        for _ in range(max_new):
            if len(gen)>=MAX_SEQ-5: break
            logits = model(torch.tensor([gen], dtype=torch.long, device=device))[0,-1]
            probs = torch.softmax(logits/temp, dim=-1)
            top30 = torch.topk(probs, 30).indices
            p = top30[random.randint(0, min(10, len(top30)-1))].item()
            gen.append(p)
            if p >= 50000: break
    return decode(gen[len(ids):])

# HTTP Chat handler
class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "vocab": 50000})
            return
        q = self.path.split("?q=")[-1] if "?q=" in self.path else "hello"
        from urllib.parse import unquote
        q = unquote(q)
        ans = respond(q)
        self._json({"question": q, "answer": ans})

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

print(f"\nChat server: http://0.0.0.0:{PORT}/?q=hello+how+are+you")
print(f"Test: curl http://localhost:{PORT}/?q=how+are+you\n")
HTTPServer(("0.0.0.0", PORT), ChatHandler).serve_forever()
