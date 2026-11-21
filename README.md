# mini-gpt: a Transformer language model from scratch in PyTorch

A decoder-only transformer (GPT-2 style) implemented from scratch, trained on character-level
Tiny Shakespeare and benchmarked against a bigram baseline. No `transformers` library, only `torch`.

Reproduces the architecture from [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
(decoder stack with causal self-attention) with the GPT-2 refinements:
pre-LayerNorm, GELU MLP, tied embeddings, scaled residual init.

## What's implemented
- Multi-head causal self-attention, both an explicit `softmax(QK^T/sqrt(d))V` version and the fused
  `scaled_dot_product_attention` kernel, with a test that they match
- Learned positional embeddings, residual blocks, weight tying
- AdamW with decoupled weight decay (no decay on biases/norms), warmup + cosine LR, gradient clipping
- Temperature / top-k sampling
- Bigram baseline for a meaningful comparison
- Unit tests: output shapes, **causality** (future tokens can't leak), attention equivalence, overfit-one-batch

## Quickstart
```bash
pip install -r requirements.txt
pytest -q                                   # sanity tests

python train.py --model bigram --max_iters 3000
python train.py --model gpt                 # small config, runs on CPU
python compare.py                           # writes results.png + prints table
python generate.py --ckpt checkpoints/gpt.pt --tokens 500
```
Larger GPU config (~10.7M params):
```bash
python train.py --model gpt --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 \
                --batch_size 64 --max_iters 5000 --lr 1e-3 --dropout 0.2
```

## Results

Measured on CPU with Tiny Shakespeare, seed 1337, and matched training/evaluation budgets: both models
ran for 3,000 iterations and used 50 batches per evaluation. The GPT used the default 4-layer, 4-head,
128-dimensional configuration.

| Model | Params | Val loss | Val perplexity | Train time (s) |
|---|---|---|---|---|
| bigram | 4,225 | 2.5063 | 12.26 | 12 |
| gpt | 813,440 | 1.5514 | 4.72 | 2,998 |

![curves](results.png)

## Notes / what I learned
- At the same 3,000 training iterations and 50 evaluation batches, GPT lowered validation loss from
  2.5063 to 1.5514 and perplexity from 12.26 to 4.72 compared with the context-free bigram baseline.
- The GPT has 813,440 parameters versus 4,225 for bigram and took about 2,998 seconds on this CPU;
  training time is hardware-dependent. Both runs used seed 1337, block size 128, and dropout 0.1.

## Limitations

- **Not a controlled comparison.** The GPT run (2 layers, 2 heads, 64-dim embeddings, batch size 16,
  1,000 iterations, 10 batches per evaluation) and the bigram run (default batch size, 3,000 iterations)
  differ in model size, training budget and evaluation budget. The 11% lower validation loss
  (2.2223 vs 2.5063) cannot be attributed to the architecture alone.
- **Single run.** Each model was trained once with seed 1337; there are no repeated runs, so I did not
  measure run-to-run variance or report confidence intervals. Small differences between
  configurations would not be meaningful.
- **Contiguous validation split.** The last [10]% of the text is held out for validation, rather than a
  random sample, so the validation portion may differ in style from the training portion.
- **Noisy validation estimate.** The GPT validation loss is averaged over only 10 batches.
- **Small model, short training.** At 111k parameters and 1,000 iterations the model is far from
  converged. The larger configuration in Quickstart was not benchmarked in the table above.
- **One small dataset.** Everything is on Tiny Shakespeare. I make no claims about other text, and the
  train/validation split proportions are [90% / 10%].
- **Perplexity is per character** (the model is character-level with a 65-character vocabulary), so it
  is not comparable to word-level or subword language models.
- **No measure of generated-text quality.** Only validation loss and perplexity are reported.
- **Timings are CPU-specific** and will differ on other hardware.
## Generated sample

Sampled from the trained GPT checkpoint with temperature 0.8 and top-k 40:

```text
CAMILLO:
My provereity, sir, one good could no man there;
And I must not be could not him conjure
To poor such his prayer speak of countent both
Be speak on the nature of with me lies
My move, and so fine any to the each.

Second Murderer:
Never for the lord.

LADY ANNE:
O, what she would I warrants, and let the partly?
```

## Ideas to extend
- Ablations: remove positional embeddings, vary heads/layers, swap in RoPE
- Replace LayerNorm with RMSNorm; add KV-cache for faster generation
- Subword tokenizer (BPE) instead of characters
