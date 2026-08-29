from dataclasses import dataclass

import pandas as pd

from inspect_capability_eval import select_samples


@dataclass
class FakeConfig:
    capability_eval_samples: int
    seed: int


def make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"instruction": [f"instruction {i}" for i in range(n)], "output": [f"output {i}" for i in range(n)]})


def test_select_samples_returns_a_prefix_of_the_real_eval_sample():
    df = make_df(50)
    config = FakeConfig(capability_eval_samples=20, seed=42)

    full_sample = df.sample(n=20, random_state=42).to_dict("records")
    selected = select_samples(df, config, num_samples=5)

    assert selected == full_sample[:5]


def test_select_samples_caps_at_available_rows():
    df = make_df(3)
    config = FakeConfig(capability_eval_samples=200, seed=42)

    selected = select_samples(df, config, num_samples=5)

    assert len(selected) == 3
