import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.config import Config
from src.prompts import format_prompt, format_text

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


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


def rouge_l_f1(prediction: str, reference: str) -> float:
    """
    ROUGE-L F-measure: longest-common-subsequence overlap instead of raw
    bag-of-words, so a reordered or lightly reworded but still faithful
    answer scores higher than plain token_f1 gives it.
    """
    if not prediction.strip() or not reference.strip():
        return float(prediction.strip() == reference.strip())
    return _ROUGE_SCORER.score(reference, prediction)["rougeL"].fmeasure


def build_chat_prompt(row: dict, tokenizer) -> str:
    """
    Build the generation prompt with the tokenizer's own ChatML template
    instead of the raw "Instruction:/Response:" string used for training
    and fidelity data collection. Every model in the grid (teacher + all
    three students) is Instruct/ChatML-tuned; prompting outside that format
    gives the model no <|im_end|> turn-end signal, so it runs to the
    generation cap instead of stopping, which tanks token-overlap-based
    capability scores regardless of answer quality. This only changes the
    capability generation prompt: fidelity is teacher-forced (no
    generation) and the on-disk training data was already collected under
    the raw format, so neither is affected.
    """
    instruction = row.get("instruction", "")
    extra_input = row.get("input", "")
    user_content = f"{instruction}\n{extra_input}" if extra_input else instruction
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def resolve_generation_eos_token_id(tokenizer) -> int:
    """
    <|im_end|> is the ChatML turn-end token generation should stop at; the
    tokenizer's own eos_token_id is the fallback for a tokenizer that
    doesn't have <|im_end|> in its vocab.
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return im_end_id if im_end_id is not None else tokenizer.eos_token_id


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

    eos_token_id = resolve_generation_eos_token_id(tokenizer)

    f1_scores = []
    rouge_scores = []
    predictions = []
    references = []

    for row in tqdm(rows, desc="Capability eval (generation)"):
        prompt = build_chat_prompt(row, tokenizer)
        # apply_chat_template's string output already contains every special
        # token the turn needs (<|im_start|>, <|im_end|>, ...), so encoding
        # it with add_special_tokens=True would risk double-adding them.
        encoded = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=config.max_length, add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        new_tokens = generated[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        reference = row.get("output", "")

        f1_scores.append(token_f1(generated_text, reference))
        rouge_scores.append(rouge_l_f1(generated_text, reference))
        predictions.append(generated_text)
        references.append(reference)

    # Scored as one batch rather than per-sample: bert_score loads its
    # scoring model once per call, so batching avoids reloading it
    # num_samples times.
    _, _, bertscore_f1 = bert_score_fn(predictions, references, lang="en", verbose=False)

    perplexity = compute_ground_truth_perplexity(model, tokenizer, rows, device)

    return {
        "token_f1": float(np.mean(f1_scores)),
        "rouge_l": float(np.mean(rouge_scores)),
        "bertscore_f1": float(bertscore_f1.mean().item()),
        "ground_truth_perplexity": perplexity,
        "num_samples": sample_size,
    }
