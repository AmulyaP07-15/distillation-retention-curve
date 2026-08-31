import yaml
from dataclasses import dataclass


@dataclass
class MeasureConfig:
    """
    Settings for the Phase 3 deployment-cost measurement (latency, memory,
    throughput, power), kept separate from the per-student training Config
    since these describe the measurement device and prompt set, not the
    student/signal being measured. One measure config = one fixed GPU, so
    switching devices means pointing --measure-config at a different file
    instead of editing student configs.
    """

    gpu_name: str = "unknown"
    device: str = "cuda"
    data_dir: str = "data"
    num_warmup_prompts: int = 5
    num_measured_prompts: int = 30
    max_new_tokens: int = 128
    batch_size: int = 1
    measure_seed: int = 123
    power_sample_interval_s: float = 0.05


def load_measure_config(config_path: str) -> MeasureConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    valid_keys = {field for field in MeasureConfig.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return MeasureConfig(**filtered)
