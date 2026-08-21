from src.capability import token_f1


def test_identical_strings_score_one():
    assert token_f1("The cat sat on the mat", "The cat sat on the mat") == 1.0


def test_disjoint_strings_score_zero():
    assert token_f1("apples and oranges", "quantum physics homework") == 0.0


def test_partial_overlap_is_between_zero_and_one():
    score = token_f1("The cat sat on the mat", "A cat sat on a mat today")
    assert 0.0 < score < 1.0


def test_case_and_punctuation_are_ignored():
    assert token_f1("Hello, world!", "hello world") == 1.0


def test_empty_prediction_against_nonempty_reference_scores_zero():
    assert token_f1("", "some reference text") == 0.0


def test_both_empty_scores_one():
    assert token_f1("", "") == 1.0
