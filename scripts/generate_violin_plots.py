"""
generate_violin_plots.py

Gera violin plots para visualizar a distribuição das métricas e
justificar o uso de testes não-paramétricos.

  - violin_quantization.pdf : desempenho (4-bit vs 8-bit)
  - violin_rag.pdf          : qualidade (com RAG vs sem RAG)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Caminhos ──────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_CSV   = PROJECT_DIR / "results" / "results_combined.csv"
OUTPUT_DIR  = PROJECT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Dados ─────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

if "success" in df.columns:
    df = df[df["success"].astype(str).str.lower() == "true"].copy()

if df["rag"].dtype != bool:
    df["rag"] = df["rag"].astype(str).str.lower().isin(["true", "1", "yes"])

df["machine_label"] = df["machine_name"].map(
    {"desktop": "Desktop", "notebook": "Notebook"}
)
df["rag_label"] = df["rag"].map({True: "com RAG", False: "sem RAG"})

# ── Estilo ────────────────────────────────────────────────────
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor":"white",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "axes.edgecolor":   "#333333",
    "axes.linewidth":   1.0,
    "grid.color":       "#DDDDDD",
    "grid.linestyle":   "-",
    "grid.linewidth":   0.7,
})

PALETTE = {"Desktop": "#1f77b4", "Notebook": "#d62728"}

def add_significance(ax, x1, x2, y, p_text, h=0.02):
    """Adiciona barra de significância entre dois grupos."""
    ymax = ax.get_ylim()[1]
    bar_y = y * ymax
    ax.plot([x1, x1, x2, x2],
            [bar_y, bar_y + h * ymax, bar_y + h * ymax, bar_y],
            lw=1.2, color="black")
    ax.text((x1 + x2) / 2, bar_y + h * ymax * 1.05,
            p_text, ha="center", va="bottom", fontsize=9)


# ══════════════════════════════════════════════════════════════
# Figura 1 — Desempenho: 4-bit vs 8-bit
# ══════════════════════════════════════════════════════════════

PERF_METRICS = [
    ("latency_s",          "Latência (s)"),
    ("throughput_tokens_s","Throughput (tok/s)"),
    ("peak_ram_mb",        "RAM pico (MB)"),
    ("peak_vram_mb",       "VRAM pico (MB)"),
]

fig1, axes1 = plt.subplots(2, 2, figsize=(11, 8))
fig1.suptitle(
    "Distribuição das métricas de desempenho por quantização",
    fontsize=13, fontweight="bold", y=1.01
)

handles1, labels1 = None, None
for ax, (metric, label) in zip(axes1.flat, PERF_METRICS):
    sns.violinplot(
        data=df,
        x="quantization",
        y=metric,
        hue="machine_label",
        ax=ax,
        palette=PALETTE,
        inner="box",
        density_norm="width",
        order=["4-bit", "8-bit"],
        hue_order=["Desktop", "Notebook"],
        linewidth=0.8,
        alpha=0.85,
    )
    ax.set_xlabel("Quantização", labelpad=6)
    ax.set_ylabel(label)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"{label}  —  Mann-Whitney p < 0,001 ***", fontsize=10)
    if handles1 is None:
        handles1, labels1 = ax.get_legend_handles_labels()
    ax.get_legend().remove()

fig1.legend(handles1, labels1, title="Máquina", loc="lower center",
            ncol=2, bbox_to_anchor=(0.5, 0), frameon=True)
fig1.tight_layout(rect=[0, 0.07, 1, 1])
out1 = OUTPUT_DIR / "violin_quantization.pdf"
fig1.savefig(out1, bbox_inches="tight")
print(f"Salvo: {out1}")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════
# Figura 2 — Qualidade: com RAG vs sem RAG
# ══════════════════════════════════════════════════════════════

QUAL_METRICS = [
    ("bertscore_f1","BERTScore F1"),
    ("rougeL_f1",   "ROUGE-L F1"),
]

RAG_PALETTE = {"Desktop": "#1f77b4", "Notebook": "#d62728"}

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
fig2.suptitle(
    "Distribuição das métricas de qualidade por uso de RAG",
    fontsize=13, fontweight="bold"
)

handles2, labels2 = None, None
for ax, (metric, label) in zip(axes2.flat, QUAL_METRICS):
    sns.violinplot(
        data=df,
        x="rag_label",
        y=metric,
        hue="machine_label",
        ax=ax,
        palette=RAG_PALETTE,
        inner="box",
        density_norm="width",
        order=["sem RAG", "com RAG"],
        hue_order=["Desktop", "Notebook"],
        linewidth=0.8,
        alpha=0.85,
    )
    ax.set_xlabel("Configuração de RAG", labelpad=6)
    ax.set_ylabel(label)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"{label}  —  Mann-Whitney p < 0,001 ***", fontsize=10)
    if handles2 is None:
        handles2, labels2 = ax.get_legend_handles_labels()
    ax.get_legend().remove()

fig2.legend(handles2, labels2, title="Máquina", loc="lower center",
            ncol=2, bbox_to_anchor=(0.5, 0), frameon=True)
fig2.tight_layout(rect=[0, 0.09, 1, 1])
out2 = OUTPUT_DIR / "violin_rag.pdf"
fig2.savefig(out2, bbox_inches="tight")
print(f"Salvo: {out2}")
plt.close(fig2)

print("\nFiguras geradas em:", OUTPUT_DIR)
