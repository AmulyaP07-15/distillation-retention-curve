import argparse
from datetime import datetime, timezone

import pandas as pd

from src.recipe import is_pareto_efficient, load_budgets, pick_winner


def format_row_line(row: dict) -> str:
    return (
        f"**{row['student']} / {row['signal']} / {row['quant']}** &mdash; "
        f"token F1 {row['token_f1']:.3f}, bertscore {row['bertscore_f1']:.3f}, "
        f"perplexity {row['ground_truth_perplexity']:.2f} | "
        f"p50 {row['p50_ms_per_token']:.2f} ms/token, "
        f"peak mem {row['peak_memory_gb']:.2f} GB, "
        f"{row['avg_power_w']:.1f} W, "
        f"{row['joules_per_token']:.3f} J/token, "
        f"{row['toks_per_sec_per_watt']:.2f} tok/s/W"
    )


def render_budget_section(budget: dict, rows: list) -> str:
    lines = [f"### {budget['name']}", "", budget.get("description", ""), ""]

    ceilings = []
    if budget.get("max_p50_ms_per_token") is not None:
        ceilings.append(f"p50 <= {budget['max_p50_ms_per_token']} ms/token")
    if budget.get("max_peak_memory_gb") is not None:
        ceilings.append(f"peak memory <= {budget['max_peak_memory_gb']} GB")
    if budget.get("max_joules_per_token") is not None:
        ceilings.append(f"energy <= {budget['max_joules_per_token']} J/token")
    if budget.get("max_avg_power_w") is not None:
        ceilings.append(f"avg power <= {budget['max_avg_power_w']} W")
    lines.append("Budget: " + (", ".join(ceilings) if ceilings else "unconstrained"))
    lines.append("")

    winner = pick_winner(rows, budget)
    if winner is None:
        lines.append("**No variant in the grid fits this budget.**")
    else:
        lines.append(f"**Winner:** {format_row_line(winner)}")

    lines.append("")
    return "\n".join(lines)


def render_pareto_table(rows: list) -> str:
    efficient_mask = is_pareto_efficient(rows)
    efficient_rows = [row for row, is_efficient in zip(rows, efficient_mask) if is_efficient]
    efficient_rows.sort(key=lambda row: row["p50_ms_per_token"])

    header = (
        "| student | signal | quant | token F1 | bertscore | perplexity | "
        "p50 ms/token | peak mem GB | J/token | tok/s/W |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, separator]
    for row in efficient_rows:
        lines.append(
            f"| {row['student']} | {row['signal']} | {row['quant']} | {row['token_f1']:.3f} | "
            f"{row['bertscore_f1']:.3f} | {row['ground_truth_perplexity']:.2f} | "
            f"{row['p50_ms_per_token']:.2f} | {row['peak_memory_gb']:.2f} | "
            f"{row['joules_per_token']:.3f} | {row['toks_per_sec_per_watt']:.2f} |"
        )
    return "\n".join(lines)


def render_recipe(df: pd.DataFrame, budgets: list, gpu_name: str) -> str:
    rows = df.to_dict("records")

    sections = [
        "# Phase 3 recipe: which student x signal x quant to deploy",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from `runs/phase3.csv`.",
        f"All latency/memory/power numbers below are from a single fixed GPU: **{gpu_name}**. "
        "Numbers from a different card are not comparable to these.",
        "",
        "## Winner per budget",
        "",
    ]

    for budget in budgets:
        sections.append(render_budget_section(budget, rows))

    sections += [
        "## Full Pareto frontier (capability vs latency vs memory vs energy)",
        "",
        "Every variant below is not beaten on every axis at once by any other variant in the grid. "
        "Pick from here directly if your budget doesn't match one of the named scenarios above.",
        "",
        render_pareto_table(rows),
        "",
    ]

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Generate RECIPE.md from runs/phase3.csv and a budgets config")
    parser.add_argument("--csv", default="runs/phase3.csv")
    parser.add_argument("--budgets", default="config/phase3_budgets.yaml")
    parser.add_argument("--out", default="RECIPE.md")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    budgets = load_budgets(args.budgets)
    gpu_name = df["gpu_name"].iloc[0] if not df.empty else "unknown"

    content = render_recipe(df, budgets, gpu_name)
    with open(args.out, "w") as f:
        f.write(content)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
