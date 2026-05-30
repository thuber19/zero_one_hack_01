import torch
from procseq.contrastive import supcon_loss


def test_supcon_lower_when_classes_separated():
    # Two tight, well-separated clusters by label -> low loss.
    sep = torch.tensor([[10.0, 0.0], [10.1, 0.0], [-10.0, 0.0], [-10.1, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    # Mixed/overlapping clusters -> higher loss.
    mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.01], [-1.0, 0.01]])
    assert supcon_loss(sep, labels) < supcon_loss(mixed, labels)


def test_supcon_no_positives_returns_zero_and_grad():
    # All distinct labels -> no same-label positives -> zero loss, graph intact.
    emb = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    loss = supcon_loss(emb, labels)
    assert float(loss.detach()) == 0.0
    loss.backward()  # must not raise


def test_supcon_is_differentiable():
    emb = torch.randn(6, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = supcon_loss(emb, labels, temperature=0.2)
    loss.backward()
    assert emb.grad is not None and torch.isfinite(loss)
