# Knowledge Distillation Retention Curves

Measuring how much of a large language model's capability survives distillation into smaller students, across three distillation signals, three model sizes, and multiple quantization levels, evaluated on two independent axes and benchmarked under a real edge-device power budget.

**Live dashboard:** https://distillation-retention-curve.streamlit.app/

## What this project answers

When you compress a 7B model into a smaller one, how much do you actually keep, and what does keeping it cost at inference time? This project builds the full pipeline to answer that with numbers rather than intuition. It distills a Qwen2.5-7B-Instruct teacher into a family of smaller students, trains them under three different distillation signals, scores every student on both fidelity to the teacher and capability on the task, then quantizes each one and measures its real latency, memory, and performance-per-watt on an NVIDIA T4. The output is a retention curve that plots capability against deployment cost, plus a recipe that says which model, signal, and quantization level to pick for a given hardware budget.

## Why it is built this way

The core idea is that fidelity and capability are not the same thing, and a compression project that measures only one of them tells half the story. A student can mimic the teacher's next-token distribution closely and still be a weak model on the actual task, or it can diverge from the teacher yet answer well. So every student is scored on two independent axes, on data splits that cannot leak into each other. Fidelity is measured on one split against the teacher. Capability is measured on a disjoint split against ground truth. Keeping these separate is what makes the retention curve honest.

## The three distillation signals

The pipeline trains students under three progressively weaker forms of supervision, so the curve shows what each level of signal buys.

**Logit-level.** The student learns from the teacher's full soft-target distribution using a temperature-scaled KL objective. This is the richest signal because it carries the teacher's uncertainty over the whole vocabulary, not just its final answer. The temperature-squared rescale on the KL term is implemented explicitly, since without it the soft-target gradient shrinks and the student barely learns.

**Sequence-level.** The student trains on the teacher's own generated outputs with plain cross-entropy, prompt tokens masked out of the loss. This is a cheaper signal that requires no stored logits, but it inherits whatever mistakes the teacher made when generating, so it confronts the honest cost of the cheaper approach.

**Trajectory-level.** A separate control track distills an action policy on a reinforcement-learning task. A PPO expert on LunarLander is cloned by a small policy, which collapses off-distribution at rollout under plain behavioral cloning, then recovers past the expert once trained with DAgger. This is the piece that shows how cloning failures appear only when the policy is actually deployed, not during training.

## Results so far

### Phase 2, the distillation grid

Six cells, three student sizes crossed with the two language signals, scored on all four metrics. Fidelity is top-1 agreement and KL divergence against the teacher. Capability is token-F1, ROUGE-L, BERTScore, and ground-truth perplexity.

| Student | Signal | Top-1 Agreement | KL Divergence | Token-F1 | ROUGE-L | BERTScore | Perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5B | logit | 0.770 | 0.349 | 0.334 | 0.266 | 0.873 | 5.52 |
| 1.5B | logit | 0.786 | 0.248 | 0.351 | 0.279 | 0.880 | 4.48 |
| 3B | logit | 0.786 | 0.286 | 0.340 | 0.269 | 0.872 | 4.18 |
| 0.5B | sequence | 0.626 | 1.270 | 0.270 | 0.207 | 0.861 | 9.07 |
| 1.5B | sequence | 0.683 | 1.477 | 0.313 | 0.247 | 0.866 | 9.96 |
| 3B | sequence | 0.728 | 0.596 | 0.305 | 0.241 | 0.863 | 6.33 |

Three findings hold up under scrutiny.

**Logit distillation dominates sequence distillation at every size, on both axes.** It wins on fidelity by a wide margin and on capability by a smaller one. If you can afford to store and train against teacher logits, that signal is the better choice here.

**Fidelity scales with capacity, but capability saturates by 1.5B.** The logit students agree with the teacher slightly more as they grow, yet their task capability is essentially flat from 1.5B to 3B once measured with enough eval samples. An apparent dip at 3B turned out to be sampling noise, confirmed by re-scoring at five times the sample count, where the gap collapsed from 0.035 to 0.011. The honest conclusion is that past 1.5B, extra capacity buys teacher-mimicry, not usefulness. That argues against reaching for the biggest model, which is exactly the kind of tradeoff a retention curve is meant to expose.

**The sequence signal improves fastest with scale.** Its fidelity climbs steeply across sizes and narrows the gap to logit as the students grow, which suggests larger students extract more from a weaker signal.

A note on evaluation. Initial token-F1 numbers were suppressed across every cell because the instruction-tuned models were being prompted without their chat template, so they had no stop signal and ran past their answers to the token cap. Diagnosing this by inspecting the actual generations, then fixing the eval to use the ChatML template with a proper end-of-turn token, raised token-F1 by thirty to forty percent relative and produced numbers that reflect real capability. BERTScore sits near 0.87 across all cells, which confirms the students produce semantically strong answers even where exact-token overlap with a single reference is low. This is why the project reports several capability metrics rather than trusting one.

### Phase 3, deployment cost under quantization

Each trained student is quantized to fp16, int8, and 4-bit NF4, then benchmarked on an NVIDIA T4, chosen deliberately because it is a roughly seventy-watt inference card, so its performance-per-watt numbers mean something for edge deployment rather than a datacenter setting. Every cell reports p50 and p99 per-token latency, peak memory, throughput, average power sampled during a sustained inference loop, tokens-per-second-per-watt, and joules-per-token, all after a discarded warmup pass and all from the same fixed device.

All 18 variants were benchmarked. The most useful result is a counterintuitive one about quantization on this card.

| Student | Signal | Quant | p50 ms/token | Peak mem GB | Perf/watt | J/token | BERTScore |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5B | logit | fp16 | 25.4 | 1.00 | 1.01 | 0.99 | 0.880 |
| 0.5B | logit | int8 | 112.5 | 0.92 | 0.29 | 3.40 | 0.879 |
| 0.5B | logit | nf4 | 38.1 | 0.77 | 0.80 | 1.25 | 0.877 |
| 1.5B | logit | fp16 | 30.0 | 3.11 | 0.49 | 2.03 | 0.876 |
| 1.5B | logit | int8 | 131.9 | 2.28 | 0.23 | 4.30 | 0.880 |
| 1.5B | logit | nf4 | 45.5 | 1.72 | 0.50 | 2.00 | 0.879 |
| 3B | logit | fp16 | 37.8 | 6.19 | 0.39 | 2.57 | 0.874 |
| 3B | logit | int8 | 167.5 | 3.45 | 0.18 | 5.71 | 0.870 |
| 3B | logit | nf4 | 57.9 | 2.25 | 0.33 | 3.04 | 0.867 |

(Sequence-signal variants track the same pattern and are in the full CSV.)

Three findings.

**Capability is essentially flat across quantization.** BERTScore stays near 0.87 and token-F1 within a few points from fp16 down to 4-bit. Compressing a student to 4-bit costs almost no capability, which is the core retention result: the quality is retained, only the cost changes.

**int8 is a trap on this card.** On the T4, bitsandbytes int8 runs roughly four times slower than fp16 and burns three to four times the energy per token, because its dequantization overhead is not offset by the T4's compute. fp16 is both the fastest and the most power-efficient option, and 4-bit NF4 sits in between. The lesson is that quantization is not automatically a win, and it has to be measured on the target hardware rather than assumed.

**NF4 is the memory play.** It cuts peak memory by roughly a quarter versus fp16 at a modest latency cost and near-identical capability, so it is the right choice when memory is the binding constraint rather than latency or power.

The recipe that falls out: use fp16 when memory allows, since it is fastest and most efficient. Drop to 4-bit NF4 when memory is tight. Avoid int8 on this class of card. That recommendation is the kind of thing a retention curve exists to produce, and it is measured rather than assumed.

## Repository layout

- `src/` core library. Teacher inference pass, the three trainers, the two-axis evaluator, quantization and deployment benchmarking, and the recipe generator.
- `run_teacher.py` builds the versioned dataset splits and the teacher logits in one pass.
- `train_student.py`, `train_student_sequence.py`, `train_trajectory.py` the three distillation paths.
- `run_eval.py` scores a checkpoint on both fidelity and capability.
- `run_phase2_grid.py` trains and evaluates the full student-by-signal grid, running the teacher pass once and reusing it across all cells.
- `run_phase3_grid.py`, `plot_phase3.py`, `generate_recipe.py` quantize, benchmark, plot, and turn the numbers into a budget-driven recipe.
- `config/` all runs are config-driven, one file per student and signal, plus the device measurement and budget configs.

## Engineering notes

The pipeline is built to run on a shared university GPU cluster under Slurm, with all training submitted as batch jobs so long runs survive disconnects. Several design decisions came out of running it at real scale rather than in a notebook.

The teacher pass runs exactly once for the whole grid rather than once per cell, since all students share the same teacher and data. Getting this right meant shelling the teacher pass out to its own subprocess so its memory is reclaimed on exit before any student trains, and adding a settings guard so a resumed run rebuilds the dataset only when the requested sample count or split actually changed.

Memory was the recurring constraint. The 3B student's optimizer state alone exceeds a 40 GB card in fp32, since AdamW holds two moment buffers plus gradients plus weights, so the fix was training that student in bfloat16 rather than reaching for a larger GPU, which is the kind of memory-budget engineering that matters on constrained hardware. Distinguishing optimizer-state memory, which scales with parameter count, from activation memory, which scales with batch size, was what pointed to the right fix.

Every distillation signal reads precomputed teacher outputs from a single versioned parquet file, so the training paths never load the teacher and never contend for GPU with it. Peak memory is reported per cell so the cost of each configuration is measured rather than assumed.

## Status

All phases are complete. Phase 1 built the pipeline and the two-axis evaluation. Phase 2 produced the distillation grid across three sizes and two signals. The trajectory-level control track is validated. Phase 3 benchmarked all eighteen quantized variants on the T4 and produced the capability-versus-cost curve and the deployment recipe. An interactive Streamlit dashboard sits on top of the two result CSVs for exploring the grid and the curve.
