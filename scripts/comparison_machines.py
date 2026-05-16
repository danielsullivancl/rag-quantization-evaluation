import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch

# ==================================================
# CONFIG
# ==================================================

INPUT_CSV = "../results/results_combined.csv"

OUTPUT_DIR = "../images_comparison"

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
# CONFIG LABELS
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

machines = [
    "desktop",
    "notebook"
]

machine_labels = {
    "desktop": "Desktop",
    "notebook": "Notebook"
}

# ==================================================
# COLORS
# ==================================================

colors = {
    "4-bit | No RAG": "#1f77b4",
    "4-bit | RAG": "#6baed6",

    "8-bit | No RAG": "#d62728",
    "8-bit | RAG": "#ff9896"
}

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
# GROUPED BARPLOT
# ==================================================

def grouped_barplot(
    ax,
    metric,
    ylabel,
    title
):

    grouped = (
        df.groupby(
            [
                "machine_name",
                "config"
            ]
        )[metric]
        .mean()
        .unstack()
        .reindex(
            machines
        )
    )

    x = np.arange(
        len(machines)
    )

    width = 0.18

    offsets = [
        -1.5 * width,
        -0.5 * width,
        0.5 * width,
        1.5 * width
    ]

    for config, offset in zip(
        order,
        offsets
    ):

        ax.bar(
            x + offset,
            grouped[config].values,
            width,
            label=config,
            color=colors[config],
            edgecolor="black"
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            machine_labels[m]
            for m in machines
        ]
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.35
    )

# ==================================================
# MEMORY PLOT
# ==================================================

def memory_plot(
    ax,
    machine
):

    subset = df[
        df["machine_name"] == machine
    ]

    grouped = (
        subset.groupby("config")
        [
            [
                "peak_vram_mb",
                "peak_ram_mb"
            ]
        ]
        .mean()
        .reindex(order)
    )

    x = np.arange(4)

    width = 0.75

    vram_values = grouped[
        "peak_vram_mb"
    ].values

    ram_values = grouped[
        "peak_ram_mb"
    ].values

    vram_colors = [
        "#1f77b4",
        "#1f77b4",
        "#d62728",
        "#d62728"
    ]

    ax.bar(
        x,
        vram_values,
        width=width,
        color=vram_colors,
        edgecolor="black",
        alpha=0.9
    )

    ax.bar(
        x,
        ram_values,
        bottom=vram_values,
        width=width,
        color=vram_colors,
        edgecolor="black",
        hatch="///",
        alpha=0.45
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        short_labels
    )

    ax.set_ylabel(
        "Peak Memory Usage (MB)"
    )

    ax.set_title(
        machine_labels[machine]
    )

    ax.grid(
        axis="y",
        alpha=0.35
    )

# ==================================================
# LATENCY COMPARISON
# ==================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

grouped_barplot(
    ax,
    "end_to_end_latency_s",
    "Latency (s)",
    "Cross-Machine Latency Comparison"
)

ax.legend()

save_figure(
    fig,
    "comparison_latency"
)

# ==================================================
# THROUGHPUT COMPARISON
# ==================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

grouped_barplot(
    ax,
    "throughput_tokens_s",
    "Tokens/s",
    "Cross-Machine Throughput Comparison"
)

ax.legend()

save_figure(
    fig,
    "comparison_throughput"
)

# ==================================================
# BERTSCORE COMPARISON
# ==================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

grouped_barplot(
    ax,
    "bertscore_f1",
    "BERTScore F1",
    "Cross-Machine BERTScore Comparison"
)

ax.legend()

save_figure(
    fig,
    "comparison_bertscore"
)

# ==================================================
# ROUGE COMPARISON
# ==================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

grouped_barplot(
    ax,
    "rougeL_f1",
    "ROUGE-L F1",
    "Cross-Machine ROUGE-L Comparison"
)

ax.legend()

save_figure(
    fig,
    "comparison_rougeL"
)

# ==================================================
# MEMORY PANEL
# ==================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

fig.suptitle(
    "Cross-Machine Memory Comparison",
    fontsize=20,
    y=1.02
)

memory_plot(
    axs[0],
    "desktop"
)

memory_plot(
    axs[1],
    "notebook"
)

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

fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.02)
)

plt.tight_layout()

save_figure(
    fig,
    "comparison_memory"
)

# ==================================================
# MAIN COMPARISON PANEL
# ==================================================

fig, axs = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

fig.suptitle(
    "Cross-Machine Comparative Analysis",
    fontsize=22,
    y=1.02
)

# --------------------------------------------------
# LATENCY
# --------------------------------------------------

grouped_barplot(
    axs[0, 0],
    "end_to_end_latency_s",
    "Latency (s)",
    "(a) Latency"
)

# --------------------------------------------------
# THROUGHPUT
# --------------------------------------------------

grouped_barplot(
    axs[0, 1],
    "throughput_tokens_s",
    "Tokens/s",
    "(b) Throughput"
)

# --------------------------------------------------
# BERTSCORE
# --------------------------------------------------

grouped_barplot(
    axs[1, 0],
    "bertscore_f1",
    "BERTScore F1",
    "(c) BERTScore"
)

# --------------------------------------------------
# ROUGE
# --------------------------------------------------

grouped_barplot(
    axs[1, 1],
    "rougeL_f1",
    "ROUGE-L F1",
    "(d) ROUGE-L"
)

handles = [
    Patch(
        facecolor=colors[c],
        edgecolor="black",
        label=label
    )
    for c, label in zip(
        order,
        short_labels
    )
]

fig.legend(
    handles=handles,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout()

save_figure(
    fig,
    "main_comparison_panel"
)

# ==================================================
# FINISHED
# ==================================================

print("\n===================================")
print("ALL COMPARISON FIGURES GENERATED")
print("===================================")

print(OUTPUT_DIR)