"""
Distillation Retention Curve — dashboard.

Run it:
    python -m pip install --upgrade streamlit plotly pandas
    python -m streamlit run dashboard.py

Put phase2_grid.csv and phase3.csv next to this file (or set paths in the sidebar).
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Distillation Retention Curve", layout="wide")

SIZE_LABEL = {"qwen0_5b": "0.5B", "qwen1_5b": "1.5B", "qwen3b": "3B"}
SIZE_PARAMS = {"qwen0_5b": 0.5, "qwen1_5b": 1.5, "qwen3b": 3.0}
QUANT_ORDER = ["fp16", "int8", "nf4"]

P3 = {
    "p50": "p50_ms_per_token", "p99": "p99_ms_per_token", "mem": "peak_memory_gb",
    "throughput": "throughput_toks_per_sec", "power": "avg_power_w",
    "perf_per_watt": "toks_per_sec_per_watt", "joules": "joules_per_token",
    "token_f1": "token_f1", "rouge_l": "rouge_l", "bertscore": "bertscore_f1",
    "ppl": "ground_truth_perplexity",
}
COST_LABEL = {
    "p50_ms_per_token": "Latency (ms per token, lower is better)",
    "peak_memory_gb": "Peak memory (GB, lower is better)",
    "joules_per_token": "Energy (joules per token, lower is better)",
    "toks_per_sec_per_watt": "Performance per watt (higher is better)",
}
CAP_LABEL = {
    "token_f1": "Token-F1", "rouge_l": "ROUGE-L",
    "bertscore_f1": "BERTScore", "ground_truth_perplexity": "Perplexity",
}


@st.cache_data
def load_csv(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str)
    return df


def with_size(df):
    df = df.copy()
    df["Size"] = df["student"].map(SIZE_LABEL).fillna(df["student"])
    df["params_b"] = df["student"].map(SIZE_PARAMS)
    return df.sort_values(["params_b", "signal"])


st.sidebar.title("Retention Curve")
st.sidebar.caption("A study of how much a large model's ability survives being compressed into a small one.")
p2 = load_csv(st.sidebar.text_input("Phase 2 CSV", "phase2_grid.csv"))
p3 = load_csv(st.sidebar.text_input("Phase 3 CSV", "phase3.csv"))

tab_intro, tab_p2, tab_p3 = st.tabs(
    ["Start here", "Which distillation method to use", "Which model and precision to ship"]
)

with tab_intro:
    st.title("How much of a big model survives when you shrink it")

    st.subheader("The problem")
    st.markdown(
        "Large language models are accurate but expensive to run. Small models are cheap "
        "but weaker. **Distillation** trains a small model to copy a large one, hoping to "
        "keep most of the ability at a fraction of the cost. The open question is simple: "
        "**how much ability do you actually keep, and what does keeping it cost when you deploy?**"
    )

    st.subheader("What this project does")
    st.markdown(
        "It takes one large teacher model (Qwen2.5-7B) and trains a family of smaller "
        "students (0.5B, 1.5B, 3B) to imitate it. Use the tabs at the top to move through "
        "the results, one question per tab."
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    for col, title, sub in [
        (c1, "1. Does it copy the teacher?", "Called fidelity. Measured against the teacher."),
        (c2, "2. Is it good at the task?", "Called capability. Measured against correct answers."),
        (c3, "3. What does it cost to run?", "Speed, memory, and power on a real edge GPU."),
    ]:
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(sub)

    st.info(
        "Key design choice: fidelity and capability are scored on separate, non-overlapping "
        "data. A model can copy the teacher closely and still be weak at the task, or the "
        "reverse. Keeping the two measurements apart is what makes the results trustworthy.",
        icon="🔑",
    )

    st.subheader("The three ways to distill")
    st.markdown(
        "- **Logit.** The student learns the teacher's full probability distribution. Richest signal.\n"
        "- **Sequence.** The student learns only from the teacher's finished answers. Cheaper, but inherits the teacher's mistakes.\n"
        "- **Trajectory.** A separate control-task track where a small policy copies an expert, fails when deployed, and is repaired with DAgger."
    )

with tab_p2:
    st.title("Which distillation method to use")
    st.caption("Comparing the two language distillation methods across three model sizes.")
    if p2 is None:
        st.info("phase2_grid.csv not found. Set the path in the sidebar.")
    else:
        df = with_size(p2)
        sig = st.multiselect("Show methods", sorted(df["signal"].unique()),
                             default=sorted(df["signal"].unique()))
        view = df[df["signal"].isin(sig)]

        st.markdown("#### The numbers")
        st.dataframe(
            view[["Size", "signal", "top1_agreement", "kl_divergence",
                  "token_f1", "rouge_l", "bertscore_f1", "ground_truth_perplexity"]]
            .rename(columns={"signal": "Method", "top1_agreement": "Copies teacher (higher better)",
                             "kl_divergence": "Diverges from teacher (lower better)",
                             "token_f1": "Token-F1", "rouge_l": "ROUGE-L",
                             "bertscore_f1": "BERTScore (meaning match)",
                             "ground_truth_perplexity": "Perplexity (lower better)"})
            .round(3),
            use_container_width=True, hide_index=True,
        )

        a, b = st.columns(2)
        with a:
            st.markdown("#### Does it copy the teacher?")
            fig = px.line(view, x="params_b", y="top1_agreement", color="signal", markers=True,
                          labels={"params_b": "Model size (billions of params)",
                                  "top1_agreement": "Agreement with teacher", "signal": "Method"})
            fig.update_layout(height=360, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Higher is better. Logit copies the teacher far more closely than sequence at every size.")
        with b:
            st.markdown("#### Is it good at the task?")
            m = st.selectbox("Metric", list(CAP_LABEL.keys())[:3],
                             format_func=lambda k: CAP_LABEL[k])
            fig2 = px.line(view, x="params_b", y=m, color="signal", markers=True,
                           labels={"params_b": "Model size (billions of params)", "signal": "Method"})
            fig2.update_layout(height=310, legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Capability barely improves past 1.5B. Bigger copies the teacher slightly better without being more useful.")

        st.success(
            "Takeaway. Logit distillation wins on both copying the teacher and task quality, "
            "at every size. Ability stops improving after 1.5B, so the smallest capable model "
            "is 1.5B, not 3B.", icon="✅",
        )

with tab_p3:
    st.title("Which model and precision to ship")
    st.caption("Every trained model, compressed to fp16, int8, and 4-bit, then measured on an NVIDIA T4 (a ~70W edge GPU).")
    if p3 is None:
        st.info("phase3.csv not found. Set the path in the sidebar.")
    else:
        df = with_size(p3)
        f1, f2 = st.columns(2)
        sig = f1.multiselect("Methods", sorted(df["signal"].unique()),
                             default=sorted(df["signal"].unique()), key="p3s")
        qs = [q for q in QUANT_ORDER if q in df["quant"].unique()]
        qsel = f2.multiselect("Precision levels", qs, default=qs)
        view = df[df["signal"].isin(sig) & df["quant"].isin(qsel)]

        st.markdown("#### Cost and quality of every version")
        cols = ["Size", "signal", "quant", P3["p50"], P3["mem"], P3["perf_per_watt"],
                P3["joules"], P3["bertscore"]]
        st.dataframe(
            view[[c for c in cols if c in view.columns]]
            .rename(columns={"signal": "Method", "quant": "Precision",
                             P3["p50"]: "Latency ms/tok", P3["mem"]: "Memory GB",
                             P3["perf_per_watt"]: "Perf/watt", P3["joules"]: "Energy J/tok",
                             P3["bertscore"]: "BERTScore"})
            .round(3),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### The retention curve: quality against cost")
        cost = st.radio("Cost to plot", list(COST_LABEL.keys()),
                        format_func=lambda k: COST_LABEL[k], horizontal=True)
        yq = st.selectbox("Quality measure", [P3["bertscore"], P3["token_f1"], P3["ppl"]],
                          format_func=lambda k: CAP_LABEL.get(k, k))
        if cost in view.columns and yq in view.columns:
            fig = px.scatter(view, x=cost, y=yq, color="Size", symbol="signal",
                             size="params_b", hover_data=["quant"],
                             labels={cost: COST_LABEL[cost], yq: CAP_LABEL.get(yq, yq),
                                     "signal": "Method"})
            fig.update_traces(marker=dict(line=dict(width=1, color="white")))
            fig.update_layout(height=520, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each dot is one version. The best versions sit toward the top-left: high quality, low cost.")

        st.success(
            "Takeaway. Quality stays almost flat across precision levels, so compression is nearly free in ability. "
            "But on this GPU int8 is a trap: about 4x slower and 3 to 4x more energy per token than fp16. "
            "Ship fp16 for speed, or 4-bit when memory is tight. Skip int8 here.", icon="✅",
        )
