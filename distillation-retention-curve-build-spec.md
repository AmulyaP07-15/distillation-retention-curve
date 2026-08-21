# Knowledge Distillation Retention Curve — Build Spec

A build spec to hand to Claude Code. It defines what to build, in what order, with an acceptance test per phase. Read the **Locked decisions** first — three of them break the project silently if wrong.

---

## Why this project (Tesla role mapping)

This project targets two Tesla AI internships. It maps most naturally to **AI Inference Co-Design** — distillation, quantization, and edge latency are its core. The **AI Infrastructure** role is covered by a clearly-marked add-on track (tagged `[INFRA]`) that grafts distributed training, profiling, and a system↔quality dashboard onto the same pipeline.

| Project piece | AI Inference Co-Design (279600) | AI Infrastructure (278578) |
|---|---|---|
| Logit / sequence / trajectory distillation | "iterate distillation techniques" | "understand training dynamics, evaluation" |
| Quantization + compression sweep | "iterate quantization/compression for edge" | — |
| Latency + peak memory + **perf-per-watt** on a fixed device | "extract max performance-per-watt", "drive latency to the minimum" | "bottlenecks in compute, networking, memory" |
| The retention curve (capability vs capacity/latency) | the perf/quality frontier | "connect system performance to model outcomes" |
| Synthetic data → retrain → re-eval | — | "data demonstrating how infra changes translate to model quality" |
| `[INFRA]` DDP multi-GPU + scaling efficiency | — | "optimize distributed training", "throughput, utilization, scalability" |
| `[INFRA]` profiler-driven bottleneck pass | — | "identify bottlenecks... using profiling/observability" |
| `[INFRA]` fused CUDA/Triton KL kernel (stretch) | CUDA | "Python and C++", "CUDA, PyTorch, comm libraries" |

Keep this table. It's also the interview script: every claim you make about the project should point back to a bullet in one of these reqs.

---

## Locked decisions (do not skip)

1. **Shared vocab is mandatory for logit distillation.** Teacher and all students must share a tokenizer/vocab or the soft-target KL is meaningless. Use the **Qwen2.5 family** — teacher `Qwen2.5-7B-Instruct`, student family `Qwen2.5-0.5B / 1.5B / 3B`. All share vocab. Decide before any code.
2. **The control task is a separate track.** Trajectory distillation shares nothing with the LLM path except the eval philosophy. Use **LunarLander-v2** with a trained **PPO expert** as the teacher policy. Cheap, reproducible, and the off-distribution collapse is visible.
3. **One fixed device for every latency/power number.** Pick the GPU now (e.g. one A100 or one 4090). Every number on the curve's x-axis comes from this device at a stated quantization. Anything else is parameter-count theater.
4. **Config-driven, versioned.** No hardcoded paths. Dataset versioning = parquet + `manifest.json` with a content hash (DVC optional). Every run reads a config and writes metrics to a run log the dashboard consumes.

### Metrics vocabulary (use Tesla's words)
- **Fidelity (vs teacher):** top-1 agreement, KL divergence, ECE.
- **Capability (vs ground truth):** task accuracy / reward.
- **System:** throughput (tok/s/GPU), GPU utilization %, MFU, `[INFRA]` scaling efficiency, convergence steps.
- **Deployment:** latency p50/p99, peak VRAM, **perf-per-watt (tok/s/W and J/tok)**.

---

## The soft-target loss (exact — Claude Code must implement this)

```
L = α · CE(student_logits, hard_labels)
  + (1 − α) · T² · KL( softmax(teacher_logits / T) ‖ softmax(student_logits / T) )
```

- The **T² multiplier** compensates the 1/T² shrinkage of the soft-target gradient. Without it the student barely learns. Start `T ∈ [2, 4]`, `α ∈ [0.1, 0.5]`.
- **Correctness gate:** a unit test asserting the soft-target gradient magnitude is stable across `T` (compare `T=1` vs `T=4` with and without the `T²` term).

---

## Phase 1 — One real vertical slice (de-risk the pipeline)

Goal: smallest student, logit path only, both eval axes plumbed, on a versioned dataset. Prove the machine end to end before scaling it.

- Repo scaffold + config system + dataset versioning (parquet + `manifest.json` + hash).
- Batched teacher-inference pass writing **logits, sequences, and trajectories** into one versioned dataset.
- Implement the temperature-scaled KL loss above, with the T² gradient test.
- Train `Qwen2.5-0.5B` on the logit path.
- Eval harness skeleton with two **disjoint** splits — fidelity on split A, capability on split B. Populate fidelity only for now.
- `[INFRA]` Wrap training in DDP from day one (even on 1 GPU) so multi-GPU is a config flag later, not a rewrite.

**Acceptance:** `train → eval` runs from config; fidelity numbers print; a test proves splits A and B do not overlap; the T² gradient test passes.

---

## Phase 2 — Breadth across signals and students

- Finish the logit path across all three students (0.5B / 1.5B / 3B).
- **Sequence-level distillation:** train on teacher-*generated* outputs only; measure how much of the teacher's mistakes get baked in. This is the honest cost of the cheaper signal.
- **Trajectory-level distillation** on LunarLander: plain behavioral cloning first, watch it collapse at rollout (off-distribution), then fix with **DAgger**. Report reward before/after the fix.
- Complete the two-axis eval on the full grid: every student × every signal, fidelity and capability, no leakage.
- `[INFRA]` Run the teacher pass and student training on 1/2/4 GPUs via DDP; record throughput, GPU utilization, MFU, and **scaling efficiency** = `throughput_N / (N · throughput_1)`.
- `[INFRA]` One **profiler pass** (torch profiler + `nsys`) over the training loop and data pipeline; identify the top bottleneck across compute / memory / data-loading and fix it. Document the before/after throughput.

**Acceptance:** a filled (3 students × 3 signals) table with fidelity + capability per cell; the trajectory reward jumps after DAgger; `[INFRA]` a scaling-efficiency plot and one documented profiler-driven speedup.

---

## Phase 3 — Enrichment, deployment realism, deliverable

- **Synthetic data at divergence regions:** find where students disagree most with the teacher (high student↔teacher KL), generate data there, retrain on the enriched mixture, re-eval. Report the capability delta — this is your "infra/data change → model quality" evidence.
- **Quantization sweep:** quantize every student (int8, 4-bit bitsandbytes, AWQ); measure **latency (p50/p99), peak VRAM, and perf-per-watt** under the fixed device budget. Sample power via `pynvml` (`nvmlDeviceGetPowerUsage`) during inference to get tok/s/W and J/tok.
- **The curve + recipe:** plot capability vs (params, latency, perf-per-watt) for all three students × three signals × quantization levels. Write the recipe the plot can't state by itself — which signal + size + quant wins per device budget.
- `[INFRA]` **Dashboard** connecting system metrics to model quality: one view where throughput / utilization / latency / watts sit beside fidelity / capability, so an infra change's effect on quality is legible at a glance. This is literally a bullet in both reqs.
- `[INFRA]` **Stretch — fused KL kernel:** a Triton (or C++/CUDA custom op) fused soft-target loss, benchmarked against the PyTorch version. This is the single highest-leverage item for the "Python and C++ / CUDA" requirement on the AI Infrastructure role. Real work — treat as optional.

**Acceptance:** the retention curve figure + a one-page recipe; a quant sweep table with latency/memory/watts; `[INFRA]` a working dashboard and, if attempted, a kernel benchmark showing speedup.

---

## Honest scope + cut lines

This is weeks of work, not a weekend, even with Claude writing the code. Prioritize by what you're actually applying to:

- **Applying mainly to Inference Co-Design (279600):** build the core (Phases 1–3 minus `[INFRA]` tags). The distillation + quantization + perf-per-watt curve *is* that job. Skip DDP/kernel/dashboard if time is tight.
- **Applying to both / mainly AI Infrastructure (278578):** you must include the `[INFRA]` track — DDP + scaling efficiency, the profiler pass, and the dashboard are what make it read for that role. The kernel is the strongest single add for the C++/CUDA bullet.
- **If "asap" means days:** ship Phase 1 + trajectory collapse/DAgger + the quantized perf-per-watt curve. Those carry both roles' signal. The full 3×3 grid is the first thing to trim.

**What you can't fake, so don't claim it:** "thousands of GPUs." You can demonstrate the *competencies* — correct DDP, measured scaling efficiency on 1–4 GPUs, MFU/utilization, a profiler-driven fix, and the system↔quality linkage. Frame it as "the methodology that scales," not "I ran it at Tesla's scale."

---

## Handoff notes for Claude Code

- Build acceptance-test-first: each phase's acceptance criteria are the definition of done.
- Keep every phase independently runnable end to end before starting the next. A half-wired pipeline that never executes top to bottom is the main failure mode.
- Config, not hardcode. One config file drives model choice, T, α, splits, GPU count, quant level.
- Log every metric (system + quality) to the run log the dashboard reads, from Phase 1 — retrofitting logging later loses the early runs you'll want on the curve.
