import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch

# ==================================================
# CONFIG
# ==================================================

INPUT_CSV = "../results/results_with_metrics.csv"


OUTPUT_DIR = "../images_desktop"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==================================================
# LOAD CSV
# ==================================================

print("Loading CSV...")

df = pd.read_csv(INPUT_CSV)

print(f"Rows: {len(df)}")

# ==================================================
# LABELS
# ==================================================

df["config"] = (
    df["quantization"]
    + " | "
    + df["rag"].map(
        {
            False: "No RAG",
            True: "RAG"
        }
    )
)

order = [
    "4-bit | No RAG",
    "4-bit | RAG",
    "8-bit | No RAG",
    "8-bit | RAG"
]

short_labels = [
    "Q4",
    "Q4 + RAG",
    "Q8",
    "Q8 + RAG"
]

# ==================================================
# COLORS
# ==================================================

colors = {
    "4-bit | No RAG": "#1f77b4",
    "4-bit | RAG": "#6baed6",

    "8-bit | No RAG": "#d62728",
    "8-bit | RAG": "#ff9896"
}

bar_colors = [
    colors[c]
    for c in order
]

# ==================================================
# STYLE
# ==================================================

plt.style.use("default")

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,

        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,

        "grid.color": "#DDDDDD",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8
    }
)

# ==================================================
# SAVE FIGURE
# ==================================================

def save_figure(fig, filename):

    path = os.path.join(
        OUTPUT_DIR,
        f"{filename}.pdf"
    )

    fig.savefig(
        path,
        bbox_inches="tight"
    )

    print(f"Saved: {path}")

# ==================================================
# CUSTOM BOXPLOT
# ==================================================

def colored_boxplot(
    ax,
    data,
    labels
):

    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        widths=0.6
    )

    for patch, color in zip(
        bp["boxes"],
        bar_colors
    ):

        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for median in bp["medians"]:

        median.set_color("black")
        median.set_linewidth(2)

    return bp

# ==================================================
# MEMORY PLOT
# ==================================================

def plot_memory_total(
    ax,
    title
):

    x = np.arange(len(order))

    width = 0.75

    # ==================================================
    # COLORS
    # ==================================================

    vram_colors = [
        "#1f77b4",  # Q4
        "#1f77b4",  # Q4 + RAG
        "#d62728",  # Q8
        "#d62728"   # Q8 + RAG
    ]

    # ==================================================
    # VRAM BARS
    # ==================================================

    vram_bars = ax.bar(
        x,
        vram_mean.values,
        width=width,
        color=vram_colors,
        edgecolor="black",
        alpha=0.9
    )

    # ==================================================
    # RAM BARS (STACKED)
    # ==================================================

    ram_bars = ax.bar(
        x,
        ram_mean.values,
        bottom=vram_mean.values,
        width=width,
        color=vram_colors,
        edgecolor="black",
        hatch="///",
        alpha=0.45
    )

    # ==================================================
    # AXES
    # ==================================================

    ax.set_xticks(x)

    ax.set_xticklabels(
        short_labels
    )

    ax.set_ylabel(
        "Peak Memory Usage (MB)"
    )

    ax.set_title(
        title
    )

    # ==================================================
    # LEGEND
    # ==================================================

    legend_elements = [

        Patch(
            facecolor="#1f77b4",
            edgecolor="black",
            label="4-bit VRAM"
        ),

        Patch(
            facecolor="#1f77b4",
            edgecolor="black",
            hatch="///",
            alpha=0.45,
            label="4-bit RAM"
        ),

        Patch(
            facecolor="#d62728",
            edgecolor="black",
            label="8-bit VRAM"
        ),

        Patch(
            facecolor="#d62728",
            edgecolor="black",
            hatch="///",
            alpha=0.45,
            label="8-bit RAM"
        )
    ]

    ax.legend(
        handles=legend_elements
    )

    # ==================================================
    # GRID
    # ==================================================

    ax.grid(
        axis="y",
        alpha=0.35
    )

# ==================================================
# PRECOMPUTED DATA
# ==================================================

latency_data = [
    df[df["config"] == c]["end_to_end_latency_s"]
    for c in order
]

throughput_data = [
    df[df["config"] == c]["throughput_tokens_s"]
    for c in order
]

bertscore_data = [
    df[df["config"] == c]["bertscore_f1"]
    for c in order
]

rouge_data = [
    df[df["config"] == c]["rougeL_f1"]
    for c in order
]

# ==================================================
# MEMORY
# ==================================================

vram_mean = (
    df.groupby("config")[
        "peak_vram_mb"
    ]
    .mean()
    .reindex(order)
)

ram_mean = (
    df.groupby("config")[
        "peak_ram_mb"
    ]
    .mean()
    .reindex(order)
)

# ==================================================
# LATENCY
# ==================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

colored_boxplot(
    ax,
    latency_data,
    short_labels
)

ax.set_ylabel(
    "Latency (s)"
)

ax.set_title(
    "End-to-End Latency"
)

ax.grid(
    alpha=0.35
)

save_figure(
    fig,
    "latency_boxplot"
)

# ==================================================
# THROUGHPUT
# ==================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

colored_boxplot(
    ax,
    throughput_data,
    short_labels
)

ax.set_ylabel(
    "Tokens/s"
)

ax.set_title(
    "Throughput"
)

ax.grid(
    alpha=0.35
)

save_figure(
    fig,
    "throughput_boxplot"
)

# ==================================================
# BERTSCORE
# ==================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

colored_boxplot(
    ax,
    bertscore_data,
    short_labels
)

ax.set_ylabel(
    "BERTScore F1"
)

ax.set_title(
    "BERTScore Evaluation"
)

ax.grid(
    alpha=0.35
)

save_figure(
    fig,
    "bertscore_boxplot"
)

# ==================================================
# ROUGE-L
# ==================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

colored_boxplot(
    ax,
    rouge_data,
    short_labels
)

ax.set_ylabel(
    "ROUGE-L F1"
)

ax.set_title(
    "ROUGE-L Evaluation"
)

ax.grid(
    alpha=0.35
)

save_figure(
    fig,
    "rougeL_boxplot"
)

# ==================================================
# MEMORY FIGURE
# ==================================================

fig, ax = plt.subplots(
    figsize=(9, 5)
)

plot_memory_total(
    ax,
    "Peak Memory Footprint"
)

save_figure(
    fig,
    "memory_footprint"
)

# ==================================================
# LATENCY VS BERTSCORE
# ==================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for config, label in zip(
    order,
    short_labels
):

    subset = df[
        df["config"] == config
    ]

    ax.scatter(
        subset["end_to_end_latency_s"],
        subset["bertscore_f1"],
        alpha=0.45,
        s=32,
        color=colors[config],
        label=label
    )

ax.set_xlabel(
    "Latency (s)"
)

ax.set_ylabel(
    "BERTScore F1"
)

ax.set_title(
    "Latency vs BERTScore"
)

ax.legend(
    frameon=True
)

ax.grid(
    alpha=0.35
)

save_figure(
    fig,
    "latency_vs_bertscore"
)

# ==================================================
# LATENCY VS ROUGE
# ==================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for config, label in zip(
    order,
    short_labels
):

    subset = df[
        df["config"] == config
    ]

    ax.scatter(
        subset["end_to_end_latency_s"],
        subset["rougeL_f1"],
        alpha=0.45,
        s=32,
        color=colors[config],
        label=label
    )

ax.set_xlabel(
    "Latency (s)"
)

ax.set_ylabel(
    "ROUGE-L F1"
)

ax.set_title(
    "Latency vs ROUGE-L"
)

ax.legend(
    frameon=True
)

ax.grid(
    alpha=0.35
)

save_figure(
    fig,
    "latency_vs_rougeL"
)

# ==================================================
# PANEL 1
# COMPUTATIONAL PERFORMANCE
# ==================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

fig.suptitle(
    "Desktop Environment - Computational Performance",
    fontsize=20,
    y=1.05
)

# --------------------------------------------------
# (a) LATENCY
# --------------------------------------------------

ax = axs[0]

colored_boxplot(
    ax,
    latency_data,
    short_labels
)

ax.set_ylabel(
    "Latency (s)"
)

ax.set_title(
    "(a) End-to-End Latency"
)

ax.grid(
    alpha=0.35
)

# --------------------------------------------------
# (b) THROUGHPUT
# --------------------------------------------------

ax = axs[1]

colored_boxplot(
    ax,
    throughput_data,
    short_labels
)

ax.set_ylabel(
    "Tokens/s"
)

ax.set_title(
    "(b) Throughput"
)

ax.grid(
    alpha=0.35
)

# --------------------------------------------------
# (c) MEMORY
# --------------------------------------------------

ax = axs[2]

plot_memory_total(
    ax,
    "(c) Peak Memory Footprint"
)

plt.tight_layout()

save_figure(
    fig,
    "panel_computational_performance"
)

# ==================================================
# PANEL 2
# RESPONSE QUALITY
# ==================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

fig.suptitle(
    "Desktop Environment - Response Quality",
    fontsize=20,
    y=1.05
)
# --------------------------------------------------
# (a) BERTSCORE
# --------------------------------------------------

ax = axs[0]

colored_boxplot(
    ax,
    bertscore_data,
    short_labels
)

ax.set_ylabel(
    "BERTScore F1"
)

ax.set_title(
    "(a) BERTScore"
)

ax.grid(
    alpha=0.35
)

# --------------------------------------------------
# (b) ROUGE-L
# --------------------------------------------------

ax = axs[1]

colored_boxplot(
    ax,
    rouge_data,
    short_labels
)

ax.set_ylabel(
    "ROUGE-L F1"
)

ax.set_title(
    "(b) ROUGE-L"
)

ax.grid(
    alpha=0.35
)

# --------------------------------------------------
# (c) LATENCY VS QUALITY
# --------------------------------------------------

ax = axs[2]

for config, label in zip(
    order,
    short_labels
):

    subset = df[
        df["config"] == config
    ]

    ax.scatter(
        subset["end_to_end_latency_s"],
        subset["bertscore_f1"],
        alpha=0.4,
        s=22,
        color=colors[config],
        label=label
    )

ax.set_xlabel(
    "Latency (s)"
)

ax.set_ylabel(
    "BERTScore F1"
)

ax.set_title(
    "(c) Latency vs Quality"
)

ax.grid(
    alpha=0.35
)

ax.legend()

plt.tight_layout()

save_figure(
    fig,
    "panel_response_quality"
)

# ==================================================
# OPTIONAL SUMMARY FIGURE
# ==================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

summary = (
    df.groupby("config")
    [
        [
            "end_to_end_latency_s",
            "bertscore_f1"
        ]
    ]
    .mean()
    .reindex(order)
)

for config, label in zip(
    order,
    short_labels
):

    ax.scatter(
        summary.loc[
            config,
            "end_to_end_latency_s"
        ],
        summary.loc[
            config,
            "bertscore_f1"
        ],
        s=260,
        color=colors[config],
        label=label
    )

ax.set_xlabel(
    "Average Latency (s)"
)

ax.set_ylabel(
    "Average BERTScore F1"
)

ax.set_title(
    "Average Latency vs Quality"
)

ax.legend()

ax.grid(
    alpha=0.35
)

save_figure(
    fig,
    "average_latency_vs_quality"
)

# ==================================================
# FINISHED
# ==================================================

print("\n===================================")
print("ALL FIGURES GENERATED")
print("===================================")

print(OUTPUT_DIR)