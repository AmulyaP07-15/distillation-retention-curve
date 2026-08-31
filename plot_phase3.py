import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical hues, one per quant variant, in the order bitsandbytes
# usually gets used (fp16 baseline -> int8 -> nf4). Validated with
# scripts/validate_palette.js from the dataviz skill (light mode, 3 slots):
# all checks pass, contrast-vs-surface warn is why every point also gets a
# direct text label instead of relying on color alone.
QUANT_COLOR = {"fp16": "#2a78d6", "int8": "#eb6834", "nf4": "#1baf7a"}
SIGNAL_MARKER = {"logit": "o", "sequence": "s"}

PLOTS = [
    ("p50_ms_per_token", "p50 latency (ms/token, lower is better)", "phase3_capability_vs_latency.png"),
    ("peak_memory_gb", "Peak GPU memory (GB, lower is better)", "phase3_capability_vs_memory.png"),
    ("joules_per_token", "Energy (J/token, lower is better)", "phase3_capability_vs_energy.png"),
]


def plot_capability_vs_cost(df: pd.DataFrame, cost_col: str, x_label: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for quant, group in df.groupby("quant"):
        for signal, sub_group in group.groupby("signal"):
            ax.scatter(
                sub_group[cost_col],
                sub_group["token_f1"],
                color=QUANT_COLOR[quant],
                marker=SIGNAL_MARKER[signal],
                s=90,
                edgecolors="black",
                linewidths=0.5,
                label=f"{quant} / {signal}",
            )

    for _, row in df.iterrows():
        ax.annotate(
            row["student"],
            (row[cost_col], row["token_f1"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#3d3d3d",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Capability: token F1 vs ground truth (higher is better)")
    ax.set_title(f"Capability vs {x_label.split(' (')[0]}")
    ax.legend(fontsize=8, title="quant / signal", loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot capability vs deployment cost from runs/phase3.csv")
    parser.add_argument("--csv", default="runs/phase3.csv")
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cost_col, x_label, filename in PLOTS:
        out_path = out_dir / filename
        plot_capability_vs_cost(df, cost_col, x_label, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
