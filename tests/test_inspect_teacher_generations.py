import pandas as pd

from inspect_teacher_generations import find_suspect_rows


def make_df(prompt_lens: list) -> pd.DataFrame:
    return pd.DataFrame({
        "prompt_index": list(range(len(prompt_lens))),
        "input_ids": [[0] * n for n in prompt_lens],
        "generated_ids": [[0] * (n + 5) for n in prompt_lens],
    })


def test_shorter_rows_in_a_batch_are_flagged():
    # batch 0: lengths [3, 7, 3] -> the two length-3 rows are shorter than the batch max (7)
    df = make_df([3, 7, 3])
    suspects = find_suspect_rows(df, batch_size=3)

    flagged_indices = {row["prompt_index"] for row in suspects}
    assert flagged_indices == {0, 2}


def test_uniform_length_batch_flags_nothing():
    df = make_df([5, 5, 5, 5])
    suspects = find_suspect_rows(df, batch_size=4)
    assert suspects == []


def test_single_row_batch_flags_nothing():
    """A batch of one has no padding at all, so it can't show the bug."""
    df = make_df([3, 8])
    suspects = find_suspect_rows(df, batch_size=1)
    assert suspects == []


def test_multiple_batches_are_handled_independently():
    # batch 0 (rows 0-1): [4, 9] -> row 0 flagged
    # batch 1 (rows 2-3): [6, 6] -> nothing flagged
    df = make_df([4, 9, 6, 6])
    suspects = find_suspect_rows(df, batch_size=2)

    flagged_indices = {row["prompt_index"] for row in suspects}
    assert flagged_indices == {0}
