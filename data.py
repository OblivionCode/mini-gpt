"""Character-level data pipeline for Tiny Shakespeare."""
import os
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path: str = "data/input.txt") -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharDataset:
    def __init__(self, text: str, val_frac: float = 0.1):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.splits = {"train": data[:n], "val": data[n:]}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        data = self.splits[split]
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)
