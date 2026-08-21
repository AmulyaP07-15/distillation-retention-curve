import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import Config
from src.prompts import format_prompt, format_text


def normalize_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    """
    Bag-of-words F1 between a generated answer and the ground-truth answer.

    Standard SQuAD-style metric: robust to word order and paraphrase length,
    unlike exact match, which is close to useless for open-ended generation.
    """
    pred_tokens = normalize_text(prediction)
    ref_tokens = normalize_text(reference)

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_ground_truth_perplexity(model, tokenizer, rows: list, device: torch.device) -> float:
    """
    Teacher-force the ground-truth response and measure how surprised the
    student is by it. This is a capability signal that does not depend on
    the teacher at all, unlike the fidelity metrics on split A.
    """
    model.eval()
    losses = []

    for row in rows:
        prompt = format_prompt(row)
        full_text = format_text(row)

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]

        if full_ids.shape[0] <= prompt_ids.shape[0]:
            continue

        labels = full_ids.clone()
        labels[: prompt_ids.shape[0]] = -100

        input_tensor = full_ids.unsqueeze(0).to(device)
        label_tensor = labels.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = label_tensor[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if torch.isfinite(loss):
            losses.append(loss.item())

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return float(np.exp(mean_loss))


def compute_capability(
    model,
    tokenizer,
    split_b_df,
    config: Config,
    device: torch.device,
) -> dict:
    """
    Task accuracy vs ground truth on split B: generate the student's own
    answer (no teacher forcing) and score it against the reference output,
    plus a teacher-forced perplexity on that same reference.
    """
    model.eval()

    sample_size = min(config.capability_eval_samples, len(split_b_df))
    rows = split_b_df.sample(n=sample_size, random_state=config.seed).to_dict("records")

    f1_scores = []

    for row in tqdm(rows, desc="Capability eval (generation)"):
        prompt = format_prompt(row)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = generated[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        f1_scores.append(token_f1(generated_text, row.get("output", "")))

    perplexity = compute_ground_truth_perplexity(model, tokenizer, rows, device)

    return {
        "token_f1": float(np.mean(f1_scores)),
        "ground_truth_perplexity": perplexity,
        "num_samples": sample_size,
    }
