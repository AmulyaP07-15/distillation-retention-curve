import yaml
from dataclasses import dataclass


@dataclass
class Config:
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct"
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    temperature: float = 2.0
    alpha: float = 0.3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    seed: int = 42
    dataset_name: str = "tatsu-lab/alpaca"
    num_samples: int = 10000
    data_dir: str = "data"
    split_a_ratio: float = 0.5
    top_k_logits: int = 1000
    capability_eval_samples: int = 200
    num_gpus: int = 1
    device: str = "cuda"
    run_log: str = "runs/run_log.jsonl"
    checkpoint_dir: str = "checkpoints/logit/qwen0_5b"
    # "float32" or "bfloat16". bfloat16 roughly halves weight + gradient +
    # AdamW optimizer-state memory (all three scale with param count, not
    # batch_size, so gradient checkpointing and a smaller batch don't touch
    # them), but needs an Ampere-or-newer GPU (A100) for hardware support.
    # V100 lacks native bf16 tensor cores, so leave smaller students on the
    # float32 default unless you know the target GPU supports it.
    student_torch_dtype: str = "float32"


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    valid_keys = {field for field in Config.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return Config(**filtered)
