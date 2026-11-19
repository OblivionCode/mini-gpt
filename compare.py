"""Plot validation curves and print a results table from checkpoints/*_log.json."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logs = [json.load(open(f)) for f in sorted(glob.glob("checkpoints/*_log.json"))]
if not logs:
    raise SystemExit("No logs found. Run train.py first.")

plt.figure(figsize=(7, 4.5))
for r in logs:
    xs = [h["iter"] for h in r["history"]]
    plt.plot(xs, [h["val_loss"] for h in r["history"]], label=f"{r['model']} (val)")
    plt.plot(xs, [h["train_loss"] for h in r["history"]], "--", alpha=0.5, label=f"{r['model']} (train)")
plt.xlabel("iteration"); plt.ylabel("cross-entropy (nats/char)")
plt.title("Tiny Shakespeare: GPT vs bigram baseline"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results.png", dpi=150)

print("| Model | Params | Val loss | Val perplexity | Train time (s) |")
print("|---|---|---|---|---|")
for r in logs:
    print(f"| {r['model']} | {r['params']:,} | {r['final_val_loss']:.4f} | "
          f"{r['val_perplexity']:.2f} | {r['train_seconds']:.0f} |")
