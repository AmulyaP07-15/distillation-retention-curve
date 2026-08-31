import yaml

CAPABILITY_COL = "token_f1"
COST_COLS = ["p50_ms_per_token", "peak_memory_gb", "joules_per_token"]

BUDGET_TO_COST_COL = {
    "max_p50_ms_per_token": "p50_ms_per_token",
    "max_peak_memory_gb": "peak_memory_gb",
    "max_joules_per_token": "joules_per_token",
    "max_avg_power_w": "avg_power_w",
}


def load_budgets(path: str) -> list:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw["budgets"]


def meets_budget(row: dict, budget: dict) -> bool:
    for budget_key, cost_col in BUDGET_TO_COST_COL.items():
        ceiling = budget.get(budget_key)
        if ceiling is not None and row[cost_col] > ceiling:
            return False
    return True


def pick_winner(rows: list, budget: dict) -> dict:
    """
    Highest capability among rows that fit the budget, tie-broken by lower
    energy per token. Returns None if nothing fits.
    """
    candidates = [row for row in rows if meets_budget(row, budget)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row[CAPABILITY_COL], -row["joules_per_token"]))


def is_pareto_efficient(rows: list) -> list:
    """
    A row is Pareto-efficient if no other row beats it on every axis at once
    (higher capability, and lower-or-equal on every cost column, with at
    least one strict improvement somewhere). Returns a same-length list of
    booleans aligned to `rows`.
    """
    efficient = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            capability_not_worse = other[CAPABILITY_COL] >= row[CAPABILITY_COL]
            costs_not_worse = all(other[col] <= row[col] for col in COST_COLS)
            strictly_better_somewhere = (other[CAPABILITY_COL] > row[CAPABILITY_COL]) or any(
                other[col] < row[col] for col in COST_COLS
            )
            if capability_not_worse and costs_not_worse and strictly_better_somewhere:
                dominated = True
                break
        efficient.append(not dominated)
    return efficient
