import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SequenceDistillationDataset(Dataset):
    """
    Sequence-level distillation: the student learns to reproduce whatever
    the teacher actually *generated*, not the soft logit distribution and
    not the ground-truth label. Cheaper to produce than top-k logits (just
    needs teacher.generate(), no full-vocab softmax), but any mistake the
    teacher made in its own generation gets trained into the student as if
    it were correct.
    """

    def __init__(self, teacher_df, tokenizer, max_length: int):
        self.teacher_df = teacher_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.teacher_df)

    def __getitem__(self, idx):
        row = self.teacher_df.iloc[idx]

        prompt_len = len(row["input_ids"])
        full_sequence = row["generated_ids"][: self.max_length]

        input_ids = torch.tensor(full_sequence, dtype=torch.long)
        seq_len = input_ids.shape[0]

        labels = torch.full((seq_len,), -100, dtype=torch.long)
        for i in range(seq_len - 1):
            next_position = i + 1
            if next_position >= prompt_len:
                labels[i] = input_ids[next_position]

        return {"input_ids": input_ids, "labels": labels}


def pad_sequence_batch(batch: list, pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)

    all_input_ids = []
    all_labels = []

    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        all_input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        all_labels.append(F.pad(item["labels"], (0, pad_len), value=-100))

    return {
        "input_ids": torch.stack(all_input_ids),
        "labels": torch.stack(all_labels),
    }
