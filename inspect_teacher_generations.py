import argparse
import random

from transformers import AutoTokenizer

from src.config import load_config
from src.dataset import load_manifest, load_teacher_logits


def find_suspect_rows(df, batch_size: int) -> list:
    """
    Rows that were shorter than the longest prompt in their original teacher-pass
    batch. Right-padded batched generation continues every row in a batch from
    the same last column, which is a pad token for anything shorter than the
    batch max, so these are exactly the rows that would show corruption if the
    right-padding bug affected this data. A row that happened to be the longest
    in its batch had no padding at all and proves nothing either way.
    """
    df = df.copy()
    df["prompt_len"] = df["input_ids"].apply(len)
    df["batch_group"] = df["prompt_index"] // batch_size

    suspects = []
    for _, group in df.groupby("batch_group"):
        if len(group) < 2:
            continue
        max_len = group["prompt_len"].max()
        suspects.extend(group[group["prompt_len"] < max_len].to_dict("records"))

    return suspects


def main():
    parser = argparse.ArgumentParser(
        description="Decode a sample of teacher_logits.parquet generations to check for right-padding corruption"
    )
    parser.add_argument("--config", default="config/students/qwen0_5b_logit.yaml", help="Path to YAML config")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size the teacher pass was actually run with (defaults to the config's batch_size)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    batch_size = args.batch_size or config.batch_size

    manifest = load_manifest(config.data_dir)
    df = load_teacher_logits(manifest)

    print(f"Loading tokenizer: {config.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)

    suspects = find_suspect_rows(df, batch_size)
    print(f"{len(suspects)} / {len(df)} rows were shorter than their batch's max prompt length")
    print("(these are the rows that would show right-padding corruption, if present)\n")

    if not suspects:
        print("No suspect rows found, nothing to sample.")
        return

    random.seed(0)
    sample = random.sample(suspects, min(args.num_samples, len(suspects)))

    for row in sample:
        prompt_text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
        full_text = tokenizer.decode(row["generated_ids"], skip_special_tokens=True)
        continuation = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text

        print("=" * 80)
        print(f"prompt_index={row['prompt_index']}  prompt_len={row['prompt_len']}")
        print(f"PROMPT      : {prompt_text[:300]!r}")
        print(f"CONTINUATION: {continuation[:500]!r}")
        print()


if __name__ == "__main__":
    main()
