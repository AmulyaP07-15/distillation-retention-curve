import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import Config
from src.dataset import load_manifest, load_teacher_logits
from src.sequence_dataset import SequenceDistillationDataset, pad_sequence_batch
from src.trainer import cleanup_ddp, log_peak_memory, reset_peak_memory_stats, resolve_torch_dtype, setup_ddp


def sequence_ce_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Plain next-token cross entropy against the teacher's own generated tokens."""
    vocab = student_logits.shape[-1]
    return F.cross_entropy(student_logits.view(-1, vocab), labels.view(-1), ignore_index=-100)


def train_sequence(rank: int, config: Config, world_size: int):
    use_ddp = world_size > 1

    if use_ddp:
        setup_ddp(rank, world_size)

    torch.manual_seed(config.seed + rank)

    device = torch.device(f"cuda:{rank}") if use_ddp else torch.device(config.device)
    reset_peak_memory_stats(device)

    manifest = load_manifest(config.data_dir)
    teacher_df = load_teacher_logits(manifest, include_per_token_columns=False)

    if rank == 0:
        print(f"Loaded {len(teacher_df)} teacher-generated rows for sequence distillation")

    tokenizer = AutoTokenizer.from_pretrained(config.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.student_model,
        torch_dtype=resolve_torch_dtype(config.student_torch_dtype),
    ).to(device)

    # Trades recompute for memory: instead of keeping every layer's activations
    # around for backward, only checkpoints are kept and the forward pass is
    # replayed layer-by-layer during backward. This is what makes the 1.5B/3B
    # sequence cells fit, since their full-vocab cross entropy (unlike the
    # logit path's top-k-bounded KL) has no smaller activation footprint to
    # fall back on. use_cache is a generation-time KV cache and doesn't help
    # (or is even incompatible with) a training forward pass, so disable it.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    dataset = SequenceDistillationDataset(teacher_df, tokenizer, config.max_length)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if use_ddp else None

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda batch: pad_sequence_batch(batch, tokenizer.pad_token_id),
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
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            loss = sequence_ce_loss(outputs.logits, labels)

            if not torch.isfinite(loss):
                # Happens when every row in the batch got truncated to max_length
                # before reaching the teacher's continuation, leaving nothing to
                # supervise. Skip rather than corrupt the model with a NaN step.
                continue

            scaled_loss = loss / config.gradient_accumulation_steps
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
                        "signal": "sequence",
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                    }
                    with open(run_log_path, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    if global_step % 20 == 0:
                        print(f"Step {global_step:5d} | sequence_ce_loss={loss.item():.4f}")

        if rank == 0:
            checkpoint_dir = Path(config.checkpoint_dir) / f"epoch_{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            underlying_model = model.module if use_ddp else model
            underlying_model.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
            print(f"Checkpoint saved: {checkpoint_dir}")

    log_peak_memory(device, rank, run_log_path)

    if use_ddp:
        cleanup_ddp()
