from src.recipe import is_pareto_efficient, meets_budget, pick_winner


def make_row(student, signal, quant, token_f1, p50, mem, joules, watts):
    return {
        "student": student,
        "signal": signal,
        "quant": quant,
        "token_f1": token_f1,
        "p50_ms_per_token": p50,
        "peak_memory_gb": mem,
        "joules_per_token": joules,
        "avg_power_w": watts,
    }


def test_meets_budget_true_when_under_every_ceiling():
    row = make_row("qwen0_5b", "logit", "nf4", 0.4, 20, 1.0, 0.3, 40)
    budget = {"max_p50_ms_per_token": 50, "max_peak_memory_gb": 2, "max_joules_per_token": 1.0}

    assert meets_budget(row, budget) is True


def test_meets_budget_false_when_over_one_ceiling():
    row = make_row("qwen3b", "logit", "fp16", 0.6, 200, 12.0, 2.0, 60)
    budget = {"max_p50_ms_per_token": 50, "max_peak_memory_gb": 2, "max_joules_per_token": 1.0}

    assert meets_budget(row, budget) is False


def test_meets_budget_ignores_unset_ceilings():
    row = make_row("qwen3b", "logit", "fp16", 0.6, 500, 12.0, 2.0, 60)
    budget = {"max_p50_ms_per_token": None, "max_peak_memory_gb": None, "max_joules_per_token": None}

    assert meets_budget(row, budget) is True


def test_pick_winner_returns_highest_capability_among_qualifying_rows():
    rows = [
        make_row("qwen0_5b", "logit", "fp16", 0.3, 20, 1.0, 0.5, 40),
        make_row("qwen0_5b", "logit", "nf4", 0.5, 25, 0.5, 0.4, 35),
        make_row("qwen3b", "logit", "fp16", 0.7, 300, 12.0, 3.0, 70),  # over budget
    ]
    budget = {"max_p50_ms_per_token": 50, "max_peak_memory_gb": 2, "max_joules_per_token": None}

    winner = pick_winner(rows, budget)

    assert winner["student"] == "qwen0_5b"
    assert winner["quant"] == "nf4"


def test_pick_winner_returns_none_when_nothing_qualifies():
    rows = [make_row("qwen3b", "logit", "fp16", 0.7, 300, 12.0, 3.0, 70)]
    budget = {"max_p50_ms_per_token": 50, "max_peak_memory_gb": None, "max_joules_per_token": None}

    assert pick_winner(rows, budget) is None


def test_pareto_efficient_excludes_a_row_strictly_worse_on_every_axis():
    dominated = make_row("qwen0_5b", "logit", "int8", 0.3, 30, 1.0, 0.5, 40)
    dominator = make_row("qwen0_5b", "logit", "nf4", 0.4, 20, 0.8, 0.3, 30)
    rows = [dominated, dominator]

    efficient = is_pareto_efficient(rows)

    assert efficient == [False, True]


def test_pareto_efficient_keeps_rows_on_the_tradeoff_frontier():
    fast_but_less_capable = make_row("qwen0_5b", "logit", "nf4", 0.3, 10, 0.5, 0.2, 30)
    slow_but_more_capable = make_row("qwen3b", "logit", "fp16", 0.7, 200, 10.0, 2.0, 60)
    rows = [fast_but_less_capable, slow_but_more_capable]

    assert is_pareto_efficient(rows) == [True, True]
