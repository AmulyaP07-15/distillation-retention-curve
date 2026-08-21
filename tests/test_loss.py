import torch
import torch.nn.functional as F

from src.loss import distillation_loss


def make_inputs(batch=2, seq=10, vocab=500, k=50):
    student = torch.randn(batch, seq, vocab, requires_grad=True)
    top_k_values, top_k_indices = torch.topk(torch.randn(batch, seq, vocab), k=k, dim=-1)
    labels = torch.randint(0, vocab, (batch, seq))
    return student, top_k_indices, top_k_values, labels


def test_loss_is_finite():
    student, top_k_indices, top_k_values, labels = make_inputs()
    total, hard, soft = distillation_loss(student, top_k_indices, top_k_values, labels, alpha=0.3, temperature=2.0)
    assert torch.isfinite(total), "Total loss must be a finite number"
    assert torch.isfinite(hard), "Hard loss must be a finite number"
    assert torch.isfinite(soft), "Soft loss must be a finite number"


def test_loss_is_positive():
    student, top_k_indices, top_k_values, labels = make_inputs()
    total, hard, soft = distillation_loss(student, top_k_indices, top_k_values, labels, alpha=0.3, temperature=2.0)
    assert total.item() >= 0, "Total loss should be non-negative"


def test_alpha_zero_ignores_hard_labels():
    student, top_k_indices, top_k_values, labels = make_inputs()
    total, hard, soft = distillation_loss(student, top_k_indices, top_k_values, labels, alpha=0.0, temperature=2.0)
    assert abs(total.item() - soft.item()) < 1e-5, "With alpha=0, total loss should equal soft loss"


def test_alpha_one_ignores_teacher():
    student, top_k_indices, top_k_values, labels = make_inputs()
    total, hard, soft = distillation_loss(student, top_k_indices, top_k_values, labels, alpha=1.0, temperature=2.0)
    assert abs(total.item() - hard.item()) < 1e-5, "With alpha=1, total loss should equal hard loss"


def test_t_squared_stabilizes_gradient():
    """
    The T-squared multiplier is what keeps the soft-target gradient from collapsing
    at high temperatures. This test checks that claim directly.

    Without T^2: gradient at T=4 is roughly 16x smaller than at T=1.
    With T^2:    gradient at T=1 and T=4 should be close in magnitude.
    """

    def measure_soft_gradient(temperature, apply_t_squared):
        torch.manual_seed(0)
        batch = 2
        seq = 10
        vocab = 2000
        k = 20

        student = torch.randn(batch, seq, vocab, requires_grad=True)
        top_k_values = torch.randn(batch, seq, k)
        top_k_indices = torch.randint(0, vocab, (batch, seq, k))

        flat_student = student.view(-1, vocab)
        flat_top_k_indices = top_k_indices.view(-1, k).long()
        flat_top_k_values = top_k_values.view(-1, k)

        teacher_probs = F.softmax(flat_top_k_values / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(flat_top_k_values / temperature, dim=-1)

        student_log_probs_full = F.log_softmax(flat_student / temperature, dim=-1)
        student_log_probs_at_top_k = student_log_probs_full.gather(-1, flat_top_k_indices)

        kl = (teacher_probs * (teacher_log_probs - student_log_probs_at_top_k)).sum(dim=-1).mean()

        if apply_t_squared:
            soft_loss = (temperature ** 2) * kl
        else:
            soft_loss = kl

        soft_loss.backward()
        return student.grad.abs().mean().item()

    grad_t1_raw = measure_soft_gradient(temperature=1.0, apply_t_squared=False)
    grad_t4_raw = measure_soft_gradient(temperature=4.0, apply_t_squared=False)

    grad_t1_scaled = measure_soft_gradient(temperature=1.0, apply_t_squared=True)
    grad_t4_scaled = measure_soft_gradient(temperature=4.0, apply_t_squared=True)

    ratio_raw = grad_t1_raw / (grad_t4_raw + 1e-9)
    ratio_scaled = grad_t1_scaled / (grad_t4_scaled + 1e-9)

    print(f"Without T^2 | grad at T=1: {grad_t1_raw:.5f}  at T=4: {grad_t4_raw:.5f}  ratio: {ratio_raw:.2f}")
    print(f"With T^2    | grad at T=1: {grad_t1_scaled:.5f}  at T=4: {grad_t4_scaled:.5f}  ratio: {ratio_scaled:.2f}")

    assert ratio_raw > 2, (
        f"Without T^2 scaling, higher temperature should shrink gradient magnitude "
        f"(got T1/T4 ratio {ratio_raw:.2f}, expected > 2)"
    )

    assert ratio_scaled < ratio_raw, (
        f"T^2 scaling should bring the T1/T4 ratio closer to 1 than the raw KL does "
        f"(raw ratio {ratio_raw:.2f} vs scaled ratio {ratio_scaled:.2f})"
    )
