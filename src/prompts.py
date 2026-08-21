def format_text(row: dict) -> str:
    """Full instruction + response text, used for teacher-forcing (logit collection, capability perplexity)."""
    instruction = row.get("instruction", "")
    extra_input = row.get("input", "")
    output = row.get("output", "")

    if extra_input:
        return f"Instruction: {instruction}\nInput: {extra_input}\nResponse: {output}"
    return f"Instruction: {instruction}\nResponse: {output}"


def format_prompt(row: dict) -> str:
    """Instruction only, no response, used when the model has to generate its own answer."""
    instruction = row.get("instruction", "")
    extra_input = row.get("input", "")

    if extra_input:
        return f"Instruction: {instruction}\nInput: {extra_input}\nResponse:"
    return f"Instruction: {instruction}\nResponse:"
