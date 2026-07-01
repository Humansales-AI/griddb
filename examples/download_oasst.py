#!/usr/bin/env python3
"""Download OpenAssistant conversations for training."""
from datasets import load_dataset
import json, random

print("Loading OpenAssistant...")
ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
qa_pairs = []
prev_q = None
for item in ds:
    if item.get("role") == "prompter":
        prev_q = item["text"].strip()
    elif item.get("role") == "assistant" and prev_q:
        a = item["text"].strip()
        if 10 < len(prev_q) < 200 and 10 < len(a) < 500:
            if "http" not in a and "```" not in a:
                qa_pairs.append((prev_q, a))
        prev_q = None
    if len(qa_pairs) >= 50000: break

print(f"Got {len(qa_pairs)} real human conversations")
with open("/workspace/oasst_chat.json", "w") as f:
    json.dump(qa_pairs, f)
print("Saved to /workspace/oasst_chat.json")
