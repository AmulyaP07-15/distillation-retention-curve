import torch

from src.sequence_dataset import SequenceDistillationDataset
from src.sequence_trainer import sequence_ce_loss


def test_loss_is_finite_and_nonnegative():
    student_logits = torch.randn(2, 10, 500, requires_grad=True)
    labels = torch.randint(0, 500, (2, 10))
    loss = sequence_ce_loss(student_logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_fully_masked_labels_give_nan_free_zero_contribution():
    """ignore_index=-100 on every position should not blow up the loss."""
    student_logits = torch.randn(2, 5, 100, requires_grad=True)
    labels = torch.full((2, 5), -100, dtype=torch.long)
    loss = sequence_ce_loss(student_logits, labels)

    assert torch.isnan(loss), "Cross entropy with no supervised positions is NaN by definition, callers must skip this batch"


def test_perfect_prediction_has_near_zero_loss():
    vocab = 20
    labels = torch.tensor([[3, 7, 1]])
    logits = torch.full((1, 3, vocab), -10.0)
    for pos, target in enumerate(labels[0]):
        logits[0, pos, target] = 10.0

    loss = sequence_ce_loss(logits, labels)
    assert loss.item() < 0.01
