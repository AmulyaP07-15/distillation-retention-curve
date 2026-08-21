from src.prompts import format_prompt, format_text


def test_format_text_includes_output():
    row = {"instruction": "Add two numbers", "input": "2 and 3", "output": "5"}
    text = format_text(row)
    assert "Add two numbers" in text
    assert "2 and 3" in text
    assert text.endswith("5")


def test_format_text_without_input_field():
    row = {"instruction": "Say hi", "input": "", "output": "Hi!"}
    text = format_text(row)
    assert "Input:" not in text
    assert text.endswith("Hi!")


def test_format_prompt_excludes_output():
    row = {"instruction": "Add two numbers", "input": "2 and 3", "output": "5"}
    prompt = format_prompt(row)
    assert "5" not in prompt
    assert prompt.endswith("Response:")


def test_format_prompt_is_a_prefix_of_format_text():
    row = {"instruction": "Say hi", "input": "", "output": "Hi!"}
    prompt = format_prompt(row)
    text = format_text(row)
    assert text.startswith(prompt)
