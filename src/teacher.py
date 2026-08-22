import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.dataset import build_and_save_splits, load_manifest, load_split, verify_manifest
from src.prompts import format_text


def run_teacher_pass(config: Config):
    manifest = load_manifest(config.data_dir)
    split_a = load_split(manifest, "split_a")

    print(f"Loading teacher model: {config.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    output_path = Path(config.data_dir) / "teacher_logits.parquet"
    records = []

    rows = split_a.to_dict("records")
    prompts = [format_text(row) for row in rows]

    for batch_start in tqdm(range(0, len(prompts), config.batch_size), desc="Teacher pass"):
        batch_prompts = prompts[batch_start : batch_start + config.batch_size]

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()

        top_values, top_indices = torch.topk(logits, k=config.top_k_logits, dim=-1)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(len(batch_prompts)):
            seq_len = int(attention_mask[i].sum().item())
            record = {
                "prompt_index": batch_start + i,
                "input_ids": input_ids[i, :seq_len].cpu().tolist(),
                "top_logit_values": top_values[i, :seq_len].cpu().tolist(),
                "top_logit_indices": top_indices[i, :seq_len].cpu().tolist(),
                "generated_ids": generated[i].cpu().tolist(),
            }
            records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(str(output_path))
    print(f"Teacher logits saved to {output_path} ({len(df)} rows)")

    manifest_path = Path(config.data_dir) / "manifest.json"
    with open(manifest_path, "r") as f:
        existing_manifest = json.load(f)

    existing_manifest["teacher_logits"] = {
        "path": str(output_path),
        "model": config.teacher_model,
        "top_k": config.top_k_logits,
        "num_rows": len(df),
    }

    with open(manifest_path, "w") as f:
        json.dump(existing_manifest, f, indent=2)

    print("Manifest updated.")


def ensure_teacher_pass(config: Config, force: bool = False) -> dict:
    """
    Build the versioned dataset splits and run the (expensive) teacher
    inference pass, but only if they don't already exist on disk. All
    students distill from the same teacher on the same data (dataset,
    num_samples, split_a_ratio, seed, top_k_logits, max_length are shared
    across every config in config/students/), so this lets the grid runner
    generate the teacher data once and have every student/signal cell
    reuse it, instead of re-running Qwen2.5-7B-Instruct once per cell.
    """
    data_dir = Path(config.data_dir)
    manifest_path = data_dir / "manifest.json"

    needs_split_rebuild = force or not manifest_path.exists()

    if not needs_split_rebuild:
        manifest = load_manifest(config.data_dir)
        stored_settings = manifest.get("config", {})
        requested_settings = {
            "dataset_name": config.dataset_name,
            "num_samples": config.num_samples,
            "split_a_ratio": config.split_a_ratio,
            "seed": config.seed,
        }

        if stored_settings != requested_settings:
            # A manifest that matches its own file hashes can still describe the
            # wrong data, e.g. someone bumped num_samples in the config since it
            # was built. Silently reusing it would eval every cell on the wrong
            # dataset without any error, so treat a settings mismatch the same as
            # missing data rather than only checking hash self-consistency.
            print(
                f"Existing manifest was built with different settings "
                f"(stored={stored_settings}, requested={requested_settings}), rebuilding."
            )
            needs_split_rebuild = True
        elif not verify_manifest(manifest):
            raise RuntimeError(
                "Dataset files do not match their stored hashes. Pass force=True to regenerate them."
            )
        else:
            print("Dataset already exists, matches the requested settings, and hashes check out.")

    if needs_split_rebuild:
        print("Building dataset splits...")
        build_and_save_splits(config)

    manifest = load_manifest(config.data_dir)
    teacher_logits_path = manifest.get("teacher_logits", {}).get("path")

    if force or needs_split_rebuild or not teacher_logits_path or not Path(teacher_logits_path).exists():
        print("Running teacher inference pass...")
        run_teacher_pass(config)
    else:
        print(f"Teacher logits already exist at {teacher_logits_path}, skipping teacher pass.")

    return load_manifest(config.data_dir)
