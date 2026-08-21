import pandas as pd

from src.sequence_dataset import SequenceDistillationDataset, pad_sequence_batch


def make_row(prompt_len, continuation_len):
    prompt = list(range(100, 100 + prompt_len))
    continuation = list(range(200, 200 + continuation_len))
    return {"input_ids": prompt, "generated_ids": prompt + continuation}


def test_prompt_positions_are_masked():
    df = pd.DataFrame([make_row(prompt_len=3, continuation_len=4)])
    dataset = SequenceDistillationDataset(df, tokenizer=None, max_length=50)

    item = dataset[0]

    # Position i's label predicts token i+1. Predicting anything inside the
    # prompt (positions 0..prompt_len-2) must stay masked.
    assert (item["labels"][:2] == -100).all()


def test_continuation_positions_are_supervised():
    df = pd.DataFrame([make_row(prompt_len=3, continuation_len=4)])
    dataset = SequenceDistillationDataset(df, tokenizer=None, max_length=50)

    item = dataset[0]
    unmasked = item["labels"][item["labels"] != -100]

    assert len(unmasked) == 4, "All 4 continuation tokens should be supervised"
    assert unmasked.tolist() == [200, 201, 202, 203]


def test_sequence_is_truncated_to_max_length():
    df = pd.DataFrame([make_row(prompt_len=3, continuation_len=10)])
    dataset = SequenceDistillationDataset(df, tokenizer=None, max_length=5)

    item = dataset[0]
    assert item["input_ids"].shape[0] == 5
    assert item["labels"].shape[0] == 5


def test_pad_sequence_batch_pads_to_longest():
    df = pd.DataFrame([make_row(3, 2), make_row(3, 5)])
    dataset = SequenceDistillationDataset(df, tokenizer=None, max_length=50)

    batch = [dataset[0], dataset[1]]
    padded = pad_sequence_batch(batch, pad_token_id=0)

    assert padded["input_ids"].shape[0] == 2
    assert padded["input_ids"].shape[1] == max(item["input_ids"].shape[0] for item in batch)
    assert padded["labels"].shape == padded["input_ids"].shape
