import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.dataset import load_manifest, load_split
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
