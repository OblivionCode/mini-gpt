"""Sample text from a trained checkpoint."""
import argparse

import torch

from model import BigramLM, GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="checkpoints/gpt.pt")
p.add_argument("--prompt", default="\n")
p.add_argument("--tokens", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
a = p.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
cfg = GPTConfig(**ck["config"])
model = (GPT(cfg) if ck["model"] == "gpt" else BigramLM(cfg))
model.load_state_dict(ck["state_dict"])
model.eval()

stoi = {c: i for i, c in enumerate(ck["chars"])}
itos = {i: c for c, i in stoi.items()}
idx = torch.tensor([[stoi[c] for c in a.prompt]], dtype=torch.long)
out = model.generate(idx, a.tokens, temperature=a.temperature, top_k=a.top_k)
print("".join(itos[int(i)] for i in out[0]))
