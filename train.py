"""Train the GPT (or the bigram baseline) on Tiny Shakespeare.

Examples:
  python train.py --model bigram --max_iters 3000
  python train.py --model gpt                       # small, CPU-friendly
  python train.py --model gpt --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 \
                  --batch_size 64 --max_iters 5000   # GPU config
"""
import argparse
import json
import math
import os
import time

import torch

from data import CharDataset, load_text
from model import BigramLM, GPT, GPTConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gpt", "bigram"], default="gpt")
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="checkpoints")
    return p.parse_args()


def lr_at(it, a):
    """Linear warmup then cosine decay to min_lr_frac * lr."""
    if it < a.warmup_iters:
        return a.lr * (it + 1) / a.warmup_iters
    progress = (it - a.warmup_iters) / max(1, a.max_iters - a.warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * coeff)


@torch.no_grad()
def estimate_loss(model, ds, a, device):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(a.eval_iters)
        for i in range(a.eval_iters):
            x, y = ds.get_batch(split, a.batch_size, a.block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"device: {device}")

    ds = CharDataset(load_text())
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=a.block_size, n_layer=a.n_layer,
                    n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout)
    model = (GPT(cfg) if a.model == "gpt" else BigramLM(cfg)).to(device)
    n_params = model.num_params()
    print(f"model: {a.model} | params: {n_params:,} | vocab: {ds.vocab_size}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.99),
    )

    os.makedirs(a.out_dir, exist_ok=True)
    history, t0 = [], time.time()
    for it in range(a.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, a)

        if it % a.eval_interval == 0 or it == a.max_iters:
            m = estimate_loss(model, ds, a, device)
            history.append({"iter": it, "train_loss": m["train"], "val_loss": m["val"]})
            print(f"iter {it:5d} | train {m['train']:.4f} | val {m['val']:.4f} "
                  f"| {time.time() - t0:.0f}s")
        if it == a.max_iters:
            break

        x, y = ds.get_batch("train", a.batch_size, a.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

    final_val = history[-1]["val_loss"]
    result = {
        "model": a.model, "params": n_params, "config": cfg.to_dict(), "args": vars(a),
        "final_val_loss": final_val, "val_perplexity": math.exp(final_val),
        "train_seconds": time.time() - t0, "history": history,
    }
    with open(os.path.join(a.out_dir, f"{a.model}_log.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                "model": a.model, "chars": ds.chars},
               os.path.join(a.out_dir, f"{a.model}.pt"))
    print(f"final val loss {final_val:.4f} (ppl {math.exp(final_val):.2f})")


if __name__ == "__main__":
    main()
