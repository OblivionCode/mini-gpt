import torch

from model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_shapes_and_loss():
    m = GPT(CFG)
    x = torch.randint(0, 20, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 20)
    assert loss.ndim == 0 and loss.item() > 0


def test_causality():
    """Changing a future token must not change earlier logits."""
    m = GPT(CFG).eval()
    a = torch.randint(0, 20, (1, 16))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 20
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)


def test_manual_attention_matches_fused():
    m = GPT(CFG).eval()
    x = torch.randint(0, 20, (2, 16))
    out_fused, _ = m(x)
    for blk in m.blocks:
        blk.attn.flash = False
    out_manual, _ = m(x)
    assert torch.allclose(out_fused, out_manual, atol=1e-4)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    m = GPT(CFG)
    x = torch.randint(0, 20, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    _, first = m(x, x)
    for _ in range(100):
        _, loss = m(x, x)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first.item() * 0.5


def test_generate_length():
    m = GPT(CFG)
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), 10)
    assert out.shape == (1, 11)
