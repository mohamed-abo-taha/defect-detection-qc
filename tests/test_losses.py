"""Unit tests for FocalLoss — these run on CPU in well under a second."""

import torch
import torch.nn.functional as F

from qc.losses import FocalLoss


def test_reduces_to_cross_entropy_when_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(16, 4)
    targets = torch.randint(0, 4, (16,))
    got = FocalLoss(gamma=0.0)(logits, targets)
    want = F.cross_entropy(logits, targets)
    assert torch.allclose(got, want, atol=1e-6)


def test_none_reduction_keeps_per_sample_shape():
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    out = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    assert out.shape == (8,)


def test_focal_never_exceeds_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(32, 5)
    targets = torch.randint(0, 5, (32,))
    fl = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    ce = F.cross_entropy(logits, targets, reduction="none")
    assert torch.all(fl <= ce + 1e-6)


def test_down_weights_easy_examples_more_than_hard():
    # easy = confident & correct; hard = barely correct
    easy = torch.tensor([[10.0, 0.0]])
    hard = torch.tensor([[0.2, 0.0]])
    t = torch.tensor([0])
    fl = FocalLoss(gamma=2.0, reduction="none")
    easy_ratio = (fl(easy, t) / F.cross_entropy(easy, t, reduction="none")).item()
    hard_ratio = (fl(hard, t) / F.cross_entropy(hard, t, reduction="none")).item()
    assert easy_ratio < hard_ratio


def test_per_class_alpha_changes_the_loss():
    torch.manual_seed(2)
    logits = torch.randn(64, 3)
    targets = torch.randint(0, 3, (64,))
    base = FocalLoss(gamma=2.0)(logits, targets)
    weighted = FocalLoss(alpha=[0.1, 1.0, 5.0], gamma=2.0)(logits, targets)
    assert not torch.allclose(base, weighted)


def test_scalar_alpha_does_not_address_imbalance():
    # documents the article's pitfall: a scalar alpha only rescales the loss,
    # leaving the per-class *balance* identical to plain focal.
    torch.manual_seed(3)
    logits = torch.randn(40, 3, requires_grad=True)
    targets = torch.randint(0, 3, (40,))
    plain = FocalLoss(alpha=None, gamma=2.0)(logits, targets)
    scaled = FocalLoss(alpha=2.0, gamma=2.0)(logits, targets)
    assert torch.allclose(scaled, 2.0 * plain, atol=1e-6)


def test_gradients_are_finite():
    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.randint(0, 4, (8,))
    FocalLoss(alpha=[1.0, 2.0, 3.0, 4.0], gamma=2.0)(logits, targets).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
