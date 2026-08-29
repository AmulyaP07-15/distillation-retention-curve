import argparse
from pathlib import Path

import torch
from bert_score import score as bert_score_fn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.capability import build_chat_prompt, resolve_generation_eos_token_id, rouge_l_f1, token_f1
from src.config import load_config
from src.dataset import load_manifest, load_split


def select_samples(split_b_df, config, num_samples: int):
    """
    The first num_samples rows of the exact same sample compute_capability()
    draws (same n, same random_state=config.seed), so these are real rows
    from the real eval, not a fresh/different sample.
    """
    sample_size = min(config.capability_eval_samples, len(split_b_df))
    rows = split_b_df.sample(n=sample_size, random_state=config.seed).to_dict("records")
    return rows[:num_samples]


def main():
    parser = argparse.ArgumentParser(
        description="Print real (prompt, student_generation, reference, metrics) tuples from the capability eval"
    )
    parser.add_argument("--config", required=True, help="Path to a student YAML config")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)

    manifest = load_manifest(config.data_dir)
    split_b_df = load_split(manifest, "split_b")
    rows = select_samples(split_b_df, config, args.num_samples)

    checkpoints = sorted(Path(config.checkpoint_dir).glob("epoch_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {config.checkpoint_dir}")
    latest_checkpoint = checkpoints[-1]
    print(f"Loading student from: {latest_checkpoint}\n")

    tokenizer = AutoTokenizer.from_pretrained(str(latest_checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(config.device)
    model = AutoModelForCausalLM.from_pretrained(str(latest_checkpoint), torch_dtype=torch.float32).to(device)
    model.eval()

    eos_token_id = resolve_generation_eos_token_id(tokenizer)

    prompts = []
    generations = []
    references = []
    hit_caps = []

    for row in rows:
        prompt = build_chat_prompt(row, tokenizer)
        reference = row.get("output", "")

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
        generation = tokenizer.decode(new_tokens, skip_special_tokens=True)

        prompts.append(prompt)
        generations.append(generation)
        references.append(reference)
        hit_caps.append(new_tokens.shape[0] >= config.max_new_tokens)

    # Batched once over all printed samples rather than per-sample, same
    # reasoning as compute_capability: bert_score's scoring model only
    # needs to load once.
    _, _, bertscore_f1s = bert_score_fn(generations, references, lang="en", verbose=False)

    for i in range(len(rows)):
        f1 = token_f1(generations[i], references[i])
        rouge_l = rouge_l_f1(generations[i], references[i])
        bertscore = bertscore_f1s[i].item()

        print(
            f"--- Sample {i} (token_f1={f1:.4f}, rougeL={rouge_l:.4f}, bertscore={bertscore:.4f}, "
            f"hit_max_cap={hit_caps[i]}) ---"
        )
        print(f"PROMPT:\n{prompts[i]}")
        print(f"\nSTUDENT GENERATION:\n{generations[i]}")
        print(f"\nREFERENCE (row['output']):\n{references[i]}")
        print()


if __name__ == "__main__":
    main()
