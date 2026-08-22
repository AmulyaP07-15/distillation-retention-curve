import pandas as pd
import torch

from src.dataset import load_teacher_logits
from src.evaluator import pad_input_ids, row_top_k_tensors
from src.sequence_dataset import SequenceDistillationDataset, pad_sequence_batch
from src.trainer import DistillationDataset, pad_batch


def make_teacher_logits_parquet(tmp_path, k=4):
    """
    Mirrors teacher.py's actual row shapes: input_ids/generated_ids are flat
    token id sequences, top_logit_values/indices are one (k,) vector per
    token position, and different rows have different sequence lengths.
    This is the shape that broke three readers one at a time in production,
    so every regression test here has to go through a real parquet
    round-trip, not synthetic in-memory objects, to actually catch it.
    """
    records = [
        {
            "prompt_index": 0,
            "input_ids": [10, 11, 12],
            "top_logit_values": [[0.1, 0.2, 0.3, 0.4], [1.1, 1.2, 1.3, 1.4], [2.1, 2.2, 2.3, 2.4]],
            "top_logit_indices": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            "generated_ids": [10, 11, 12, 20, 21],
        },
        {
            "prompt_index": 1,
            "input_ids": [30, 31],
            "top_logit_values": [[3.1, 3.2, 3.3, 3.4], [4.1, 4.2, 4.3, 4.4]],
            "top_logit_indices": [[13, 14, 15, 16], [17, 18, 19, 20]],
            "generated_ids": [30, 31, 40, 41, 42, 43],
        },
    ]
    df = pd.DataFrame(records)
    path = tmp_path / "teacher_logits.parquet"
    df.to_parquet(str(path))
    return {"teacher_logits": {"path": str(path)}}


def test_load_teacher_logits_normalizes_sequence_columns(tmp_path):
    manifest = make_teacher_logits_parquet(tmp_path)
    df = load_teacher_logits(manifest)

    assert isinstance(df.iloc[0]["input_ids"], list)
    assert isinstance(df.iloc[0]["generated_ids"], list)


def test_load_teacher_logits_normalizes_per_token_columns_to_2d_arrays(tmp_path):
    manifest = make_teacher_logits_parquet(tmp_path, k=4)
    df = load_teacher_logits(manifest)

    row0 = df.iloc[0]
    assert row0["top_logit_indices"].shape == (3, 4)
    assert row0["top_logit_values"].shape == (3, 4)

    row1 = df.iloc[1]
    assert row1["top_logit_indices"].shape == (2, 4)
    assert row1["top_logit_values"].shape == (2, 4)


def test_distillation_dataset_batches_variable_length_rows(tmp_path):
    manifest = make_teacher_logits_parquet(tmp_path)
    df = load_teacher_logits(manifest)

    dataset = DistillationDataset(df, tokenizer=None, max_length=512)
    batch = [dataset[0], dataset[1]]
    padded = pad_batch(batch, pad_token_id=0)

    max_len = max(len(row) for row in df["input_ids"])
    assert padded["input_ids"].shape == (2, max_len)
    assert padded["top_k_indices"].shape == (2, max_len, 4)
    assert padded["top_k_values"].shape == (2, max_len, 4)
    assert padded["position_mask"].shape == (2, max_len)

    # Row 1 is shorter, so its tail should be masked out
    assert padded["position_mask"][1].tolist() == [True, True, False]


def test_fidelity_padding_and_top_k_tensors_survive_the_roundtrip(tmp_path):
    manifest = make_teacher_logits_parquet(tmp_path)
    df = load_teacher_logits(manifest)
    rows = df.to_dict("records")

    padded_ids = pad_input_ids(rows, pad_token_id=0)
    assert padded_ids[0] == [10, 11, 12]
    assert padded_ids[1] == [30, 31, 0]

    top_k_indices, top_k_values = row_top_k_tensors(rows[0])
    assert top_k_indices.shape == (3, 4)
    assert top_k_values.shape == (3, 4)
    assert torch.equal(top_k_indices[0], torch.tensor([1, 2, 3, 4]))


def test_sequence_dataset_reads_generated_ids_after_roundtrip(tmp_path):
    manifest = make_teacher_logits_parquet(tmp_path)
    df = load_teacher_logits(manifest)

    dataset = SequenceDistillationDataset(df, tokenizer=None, max_length=512)
    batch = [dataset[0], dataset[1]]
    padded = pad_sequence_batch(batch, pad_token_id=0)

    assert padded["input_ids"].shape[0] == 2
    assert padded["labels"].shape == padded["input_ids"].shape
