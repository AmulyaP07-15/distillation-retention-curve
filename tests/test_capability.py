from src.capability import build_chat_prompt, resolve_generation_eos_token_id, rouge_l_f1, token_f1


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer, so chat-prompt/eos logic can be
    unit tested without downloading a real Qwen tokenizer."""

    def __init__(self, vocab=None, eos_token_id=999):
        self.vocab = vocab or {}
        self.eos_token_id = eos_token_id

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)


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


def test_rouge_l_identical_strings_score_one():
    assert rouge_l_f1("The cat sat on the mat", "The cat sat on the mat") == 1.0


def test_rouge_l_disjoint_strings_score_zero():
    assert rouge_l_f1("apples and oranges", "quantum physics homework") == 0.0


def test_rouge_l_rewarded_word_order_more_than_bag_of_words():
    # Same words, correct order for pred vs reordered scramble for scramble.
    pred = "the cat sat on the mat today"
    reference = "the cat sat on the mat today"
    scrambled_reference = "today mat the on sat cat the"

    assert rouge_l_f1(pred, reference) > rouge_l_f1(pred, scrambled_reference)


def test_rouge_l_empty_prediction_against_nonempty_reference_scores_zero():
    assert rouge_l_f1("", "some reference text") == 0.0


def test_build_chat_prompt_uses_instruction_only_when_no_input():
    row = {"instruction": "Summarize this.", "input": "", "output": "A summary."}
    prompt = build_chat_prompt(row, FakeTokenizer())

    assert "Summarize this." in prompt
    assert "<|im_start|>user" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_build_chat_prompt_folds_input_into_user_message():
    row = {"instruction": "Translate this.", "input": "Bonjour", "output": "Hello"}
    prompt = build_chat_prompt(row, FakeTokenizer())

    assert "Translate this.\nBonjour" in prompt


def test_resolve_generation_eos_token_id_prefers_im_end():
    tokenizer = FakeTokenizer(vocab={"<|im_end|>": 7}, eos_token_id=999)
    assert resolve_generation_eos_token_id(tokenizer) == 7


def test_resolve_generation_eos_token_id_falls_back_when_im_end_missing():
    tokenizer = FakeTokenizer(vocab={}, eos_token_id=999)
    assert resolve_generation_eos_token_id(tokenizer) == 999
