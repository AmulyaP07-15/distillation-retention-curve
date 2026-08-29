import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.capability import normalize_text, token_f1
from src.config import load_config
from src.dataset import load_manifest, load_split
from src.prompts import format_prompt


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
        description="Print real (prompt, student_generation, reference) triples from the capability eval"
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

    for i, row in enumerate(rows):
        prompt = format_prompt(row)
        reference = row.get("output", "")

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
        generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        hit_token_cap = new_tokens.shape[0] >= 128

        f1 = token_f1(generation, reference)

        print(f"--- Sample {i} (F1={f1:.4f}, hit_128_token_cap={hit_token_cap}) ---")
        print(f"PROMPT:\n{prompt}")
        print(f"\nSTUDENT GENERATION ({new_tokens.shape[0]} tokens):\n{generation}")
        print(f"\nREFERENCE (row['output']):\n{reference}")
        print(f"\nnormalized pred tokens : {normalize_text(generation)[:20]}")
        print(f"normalized ref tokens  : {normalize_text(reference)[:20]}")
        print()


if __name__ == "__main__":
    main()
