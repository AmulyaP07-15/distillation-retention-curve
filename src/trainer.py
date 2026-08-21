import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import Config
from src.dataset import load_manifest, load_split
from src.loss import distillation_loss


class DistillationDataset(Dataset):
    def __init__(self, teacher_df: pd.DataFrame, tokenizer, max_length: int):
        self.teacher_df = teacher_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.teacher_df)

    def __getitem__(self, idx):
        row = self.teacher_df.iloc[idx]

        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        top_k_indices = torch.tensor(row["top_logit_indices"], dtype=torch.long)
        top_k_values = torch.tensor(row["top_logit_values"], dtype=torch.float)

        # Shift one position left so label[i] is the token the model at position i predicts
        labels = torch.cat([input_ids[1:], torch.tensor([-100])])

        return {
            "input_ids": input_ids,
            "top_k_indices": top_k_indices,
            "top_k_values": top_k_values,
            "labels": labels,
        }


def pad_batch(batch: list, pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    k = batch[0]["top_k_indices"].shape[1]

    all_input_ids = []
    all_top_k_indices = []
    all_top_k_values = []
    all_labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        all_input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        all_top_k_indices.append(F.pad(item["top_k_indices"], (0, 0, 0, pad_len), value=0))
        all_top_k_values.append(F.pad(item["top_k_values"], (0, 0, 0, pad_len), value=-1e9))
        all_labels.append(F.pad(item["labels"], (0, pad_len), value=-100))

    return {
        "input_ids": torch.stack(all_input_ids),
        "top_k_indices": torch.stack(all_top_k_indices),
        "top_k_values": torch.stack(all_top_k_values),
        "labels": torch.stack(all_labels),
    }


def setup_ddp(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def train(rank: int, config: Config, world_size: int):
    use_ddp = world_size > 1

    if use_ddp:
        setup_ddp(rank, world_size)

    torch.manual_seed(config.seed + rank)

    device = torch.device(f"cuda:{rank}") if use_ddp else torch.device(config.device)

    manifest = load_manifest(config.data_dir)
    teacher_df = pd.read_parquet(manifest["teacher_logits"]["path"])

    if rank == 0:
        print(f"Loaded {len(teacher_df)} teacher rows from parquet")

    tokenizer = AutoTokenizer.from_pretrained(config.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.student_model,
        torch_dtype=torch.float32,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    dataset = DistillationDataset(teacher_df, tokenizer, config.max_length)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if use_ddp else None

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda batch: pad_batch(batch, tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = (len(loader) // config.gradient_accumulation_steps) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    run_log_path = Path(config.run_log)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(config.num_epochs):
        if use_ddp:
            sampler.set_epoch(epoch)

        model.train()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            top_k_indices = batch["top_k_indices"].to(device)
            top_k_values = batch["top_k_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            student_logits = outputs.logits

            total_loss, hard_loss, soft_loss = distillation_loss(
                student_logits,
                top_k_indices,
                top_k_values,
                labels,
                alpha=config.alpha,
                temperature=config.temperature,
            )

            scaled_loss = total_loss / config.gradient_accumulation_steps
            scaled_loss.backward()

            should_step = (batch_idx + 1) % config.gradient_accumulation_steps == 0

            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0:
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "total_loss": total_loss.item(),
                        "hard_loss": hard_loss.item(),
                        "soft_loss": soft_loss.item(),
                    }
                    with open(run_log_path, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    if global_step % 20 == 0:
                        print(
                            f"Step {global_step:5d} | "
                            f"total={total_loss.item():.4f}  "
                            f"hard={hard_loss.item():.4f}  "
                            f"soft={soft_loss.item():.4f}"
                        )

        if rank == 0:
            checkpoint_dir = Path(config.checkpoint_dir) / f"epoch_{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            underlying_model = model.module if use_ddp else model
            underlying_model.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
            print(f"Checkpoint saved: {checkpoint_dir}")

    if use_ddp:
        cleanup_ddp()
