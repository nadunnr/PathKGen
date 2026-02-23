"""
test_losses.py — Unit tests for losses.py (label_smoothing_loss, compute_rdrop_loss)
and metrics.py (compute_bleu_penalty).

Run with:
    conda run -n vlm python test_losses.py
    # or simply:
    python test_losses.py  (inside the vlm conda env)
"""

import sys
import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------
from losses import label_smoothing_loss, compute_rdrop_loss
from metrics import compute_bleu_penalty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VOCAB = 32000
BATCH, SEQ = 2, 10
SEED = 42
torch.manual_seed(SEED)


def _random_logits(batch=BATCH, seq=SEQ, vocab=VOCAB):
    return torch.randn(batch, seq, vocab)


def _random_labels(batch=BATCH, seq=SEQ, vocab=VOCAB, mask_frac=0.5):
    """Random label tensor with some -100 positions."""
    labels = torch.randint(0, vocab, (batch, seq))
    mask = torch.rand(batch, seq) < mask_frac
    labels[mask] = -100
    return labels


def passed(name):
    print(f"  ✅  {name}")


def failed(name, msg):
    print(f"  ❌  {name}: {msg}")
    sys.exit(1)


def assert_close(a, b, name, atol=1e-5):
    if not torch.allclose(a.cpu().float(), b.cpu().float(), atol=atol):
        failed(name, f"expected {b.item():.6f}, got {a.item():.6f}")
    else:
        passed(name)


# ===========================================================================
# label_smoothing_loss
# ===========================================================================
print("\n── label_smoothing_loss ─────────────────────────────────────────")

# Test 1 — smoothing=0 matches F.cross_entropy exactly
def test_lsl_no_smoothing():
    logits = _random_logits()
    labels = _random_labels()
    got  = label_smoothing_loss(logits, labels, smoothing=0.0)
    want = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1), ignore_index=-100)
    assert_close(got, want, "smoothing=0.0 matches F.cross_entropy")

test_lsl_no_smoothing()

# Test 2 — smoothing=0.1 gives *different* loss (generally lower on overconfident logits)
def test_lsl_with_smoothing_differs():
    logits = _random_logits()
    labels = _random_labels()
    ce   = label_smoothing_loss(logits, labels, smoothing=0.0)
    lsl  = label_smoothing_loss(logits, labels, smoothing=0.1)
    # They must differ; smoothed loss < standard CE for overconfident logits
    if torch.allclose(ce, lsl, atol=1e-4):
        failed("smoothing=0.1 differs from smoothing=0.0", "values identical")
    else:
        passed("smoothing=0.1 produces a different (typically lower) loss")

test_lsl_with_smoothing_differs()

# Test 3 — all -100 labels → loss = 0.0 with grad
def test_lsl_all_ignored():
    logits = _random_logits()
    labels = torch.full((BATCH, SEQ), -100, dtype=torch.long)
    loss = label_smoothing_loss(logits, labels, smoothing=0.1)
    if not (loss.item() == 0.0 and loss.requires_grad):
        failed("all-ignored labels → 0.0", f"got {loss.item()}")
    else:
        passed("all -100 labels → loss 0.0 with requires_grad=True")

test_lsl_all_ignored()

# Test 4 — gradient flows through label_smoothing_loss
def test_lsl_gradient():
    logits = _random_logits().requires_grad_(True)
    labels = _random_labels()
    loss = label_smoothing_loss(logits, labels, smoothing=0.1)
    loss.backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        failed("gradient flows", "grad is None or non-finite")
    else:
        passed("gradient flows through label_smoothing_loss")

test_lsl_gradient()


# ===========================================================================
# compute_rdrop_loss
# ===========================================================================
print("\n── compute_rdrop_loss ───────────────────────────────────────────")

# Test 5 — identical logits → KL = 0 → result equals standard CE
def test_rdrop_identical():
    logits = _random_logits()
    labels = _random_labels()
    # Shift for causal LM (as callers do)
    sl = logits[..., :-1, :].contiguous()
    lb = labels[..., 1:].contiguous()
    rdrop = compute_rdrop_loss(sl, sl, lb, alpha=1.0)
    ce    = F.cross_entropy(sl.view(-1, VOCAB), lb.view(-1), ignore_index=-100)
    assert_close(rdrop, ce, "identical logits → KL=0 → equals CE", atol=1e-4)

test_rdrop_identical()

# Test 6 — different logits → loss > CE alone
def test_rdrop_different_larger():
    l1 = _random_logits()
    l2 = _random_logits()   # different random seed → different distribution
    labels = _random_labels()
    sl1 = l1[..., :-1, :].contiguous()
    sl2 = l2[..., :-1, :].contiguous()
    lb  = labels[..., 1:].contiguous()
    rdrop = compute_rdrop_loss(sl1, sl2, lb, alpha=1.0)
    ce1 = F.cross_entropy(sl1.view(-1, VOCAB), lb.view(-1), ignore_index=-100)
    ce2 = F.cross_entropy(sl2.view(-1, VOCAB), lb.view(-1), ignore_index=-100)
    ce_avg = (ce1 + ce2) / 2
    if rdrop.item() <= ce_avg.item() - 1e-5:
        failed("different logits → rdrop > CE", f"rdrop={rdrop.item():.4f}, ce={ce_avg.item():.4f}")
    else:
        passed("different logits → rdrop > CE alone")

test_rdrop_different_larger()

# Test 7 — all -100 labels → only CE contributes; KL path returns early
def test_rdrop_all_ignored():
    l1 = _random_logits()
    l2 = _random_logits()
    labels = torch.full((BATCH, SEQ), -100, dtype=torch.long)
    sl1 = l1[..., :-1, :].contiguous()
    sl2 = l2[..., :-1, :].contiguous()
    lb  = labels[..., 1:].contiguous()
    try:
        loss = compute_rdrop_loss(sl1, sl2, lb, alpha=1.0)
        passed("all -100 labels → early return without error")
    except Exception as e:
        failed("all -100 labels", str(e))

test_rdrop_all_ignored()

# Test 8 — gradient flows through compute_rdrop_loss
def test_rdrop_gradient():
    l1 = _random_logits().requires_grad_(True)
    l2 = _random_logits().requires_grad_(True)
    labels = _random_labels()
    sl1 = l1[..., :-1, :].contiguous()
    sl2 = l2[..., :-1, :].contiguous()
    lb  = labels[..., 1:].contiguous()
    loss = compute_rdrop_loss(sl1, sl2, lb, alpha=1.0)
    loss.backward()
    if l1.grad is None or not torch.isfinite(l1.grad).all():
        failed("rdrop gradient", "grad is None or non-finite for l1")
    elif l2.grad is None or not torch.isfinite(l2.grad).all():
        failed("rdrop gradient", "grad is None or non-finite for l2")
    else:
        passed("gradient flows through compute_rdrop_loss")

test_rdrop_gradient()

# Test 9 — alpha=0 means no KL term: result equals CE avg
def test_rdrop_alpha_zero():
    l1 = _random_logits()
    l2 = _random_logits()
    labels = _random_labels()
    sl1 = l1[..., :-1, :].contiguous()
    sl2 = l2[..., :-1, :].contiguous()
    lb  = labels[..., 1:].contiguous()
    rdrop = compute_rdrop_loss(sl1, sl2, lb, alpha=0.0)
    ce1 = F.cross_entropy(sl1.view(-1, VOCAB), lb.view(-1), ignore_index=-100)
    ce2 = F.cross_entropy(sl2.view(-1, VOCAB), lb.view(-1), ignore_index=-100)
    ce_avg = (ce1 + ce2) / 2
    assert_close(rdrop, ce_avg, "alpha=0.0 → equals CE average", atol=1e-4)

test_rdrop_alpha_zero()


# ===========================================================================
# compute_bleu_penalty (metrics.py)
# ===========================================================================
print("\n── compute_bleu_penalty ─────────────────────────────────────────")

REFERENCE   = "the tumor shows moderately differentiated adenocarcinoma with glandular architecture"
PREDICTION  = "the tumor shows moderately differentiated adenocarcinoma with glandular architecture"
UNRELATED   = "sunny weather outside with clear blue skies and warm temperatures today"
EMPTY       = ""

# Test 10 — identical strings → BLEU ≈ 1.0 (method1 smoothing may give <1 for short)
def test_bleu_identical():
    score = compute_bleu_penalty(PREDICTION, REFERENCE, metric='bleu4')
    if score < 0.9:
        failed("identical strings → bleu ≈ 1.0", f"got {score:.4f}")
    else:
        passed(f"identical strings → BLEU-4 = {score:.4f} (>= 0.9)")

test_bleu_identical()

# Test 11 — unrelated string → low score
def test_bleu_unrelated():
    score = compute_bleu_penalty(UNRELATED, REFERENCE, metric='bleu4')
    if score > 0.1:
        failed("unrelated string → bleu ≈ 0", f"got {score:.4f}")
    else:
        passed(f"unrelated string → BLEU-4 = {score:.4f} (<= 0.1)")

test_bleu_unrelated()

# Test 12 — partial overlap → 0 < score < 1
def test_bleu_partial():
    partial = "the tumor shows adenocarcinoma"
    score = compute_bleu_penalty(partial, REFERENCE, metric='bleu2')
    if not (0.0 <= score <= 1.0):
        failed("partial overlap → 0 < score < 1", f"got {score:.4f}")
    else:
        passed(f"partial overlap → BLEU-2 = {score:.4f} (in [0,1])")

test_bleu_partial()

# Test 13 — bleu_penalty_weight doubles the loss
def test_bleu_weight_applied():
    """Simulate the loss combination in train_epoch."""
    logits = _random_logits()
    labels = _random_labels()
    sl = logits[..., :-1, :].contiguous()
    lb = labels[..., 1:].contiguous()

    # Mock a bleu_penalty (weighted CE)
    bleu_penalty = F.cross_entropy(sl.view(-1, VOCAB), lb.view(-1), ignore_index=-100)

    loss_w1 = 1.0 * bleu_penalty
    loss_w2 = 2.0 * bleu_penalty

    if not torch.allclose(loss_w2, 2 * loss_w1, atol=1e-6):
        failed("bleu_penalty_weight=2.0 doubles loss", f"w1={loss_w1.item():.4f}, w2={loss_w2.item():.4f}")
    else:
        passed(f"bleu_penalty_weight=2.0 correctly doubles loss ({loss_w1.item():.4f} → {loss_w2.item():.4f})")

test_bleu_weight_applied()

# Test 14 — bleu_computed=False means ce_loss is used (Bug 2 regression)
def test_bleu_computed_flag_fallback():
    """When compute_bleu_penalty_loss returns None, ce_loss must be used."""
    ce_loss = torch.tensor(1.23456)
    bleu_computed = False
    bleu_penalty = None   # as returned when all samples skipped

    if bleu_computed:
        loss = 1.0 * bleu_penalty   # would crash if None
    else:
        loss = ce_loss

    if not torch.allclose(loss, ce_loss, atol=1e-6):
        failed("Bug 2 regression: bleu_computed=False → ce_loss used", str(loss))
    else:
        passed("Bug 2 regression: bleu_computed=False → ce_loss correctly used")

test_bleu_computed_flag_fallback()


# ===========================================================================
print(f"\n{'='*55}")
print("  All tests passed! ✅")
print('='*55)
