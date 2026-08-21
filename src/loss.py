import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    top_k_indices: torch.Tensor,
    top_k_values: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float,
) -> tuple:
    """
    Combine hard-label cross entropy with a sparse soft-target KL term.

    We store only the teacher's top-k logits per position, so instead of
    computing KL over the full vocabulary we compute it just over those k
    positions. The teacher's near-zero probability on the remaining tokens
    makes their contribution negligible anyway.

    The T-squared multiplier restores the gradient magnitude that gets
    compressed when you divide logits by temperature before softmax.
    Without it the student barely learns from the soft targets.

    student_logits : (batch, seq, vocab)
    top_k_indices  : (batch, seq, k)  teacher top-k token positions
    top_k_values   : (batch, seq, k)  teacher top-k logit values
    labels         : (batch, seq)     next-token hard labels (-100 to ignore)
    """

    batch, seq, vocab = student_logits.shape
    k = top_k_indices.shape[-1]

    flat_student = student_logits.view(-1, vocab)
    flat_top_k_indices = top_k_indices.view(-1, k).long()
    flat_top_k_values = top_k_values.view(-1, k)
    flat_labels = labels.view(-1)

    hard_loss = F.cross_entropy(flat_student, flat_labels, ignore_index=-100)

    teacher_probs = F.softmax(flat_top_k_values / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(flat_top_k_values / temperature, dim=-1)

    student_log_probs_full = F.log_softmax(flat_student / temperature, dim=-1)
    student_log_probs_at_top_k = student_log_probs_full.gather(-1, flat_top_k_indices)

    kl = (teacher_probs * (teacher_log_probs - student_log_probs_at_top_k)).sum(dim=-1).mean()

    soft_loss = (temperature ** 2) * kl

    total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss

    return total_loss, hard_loss.detach(), soft_loss.detach()
