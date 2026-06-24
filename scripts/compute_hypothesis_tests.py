"""
compute_hypothesis_tests.py

Aplica o teste de Mann-Whitney U para as duas comparações principais:
  1. Quantização: 4-bit vs 8-bit (métricas de desempenho)
  2. RAG: com RAG vs sem RAG (métricas de qualidade)

Saída: results/hypothesis_mannwhitney.csv e tabela LaTeX.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# ── Caminhos ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_CSV  = PROJECT_DIR / "results" / "results_combined.csv"
OUTPUT_CSV = PROJECT_DIR / "results" / "hypothesis_mannwhitney.csv"
OUTPUT_TEX = PROJECT_DIR / "results" / "hypothesis_mannwhitney.tex"

# ── Configuração ──────────────────────────────────────────────
ALPHA = 0.05

PERFORMANCE_METRICS = {
    "latency_s":          "Latencia (s)",
    "throughput_tokens_s":"Throughput (tok/s)",
    "peak_ram_mb":        "RAM pico (MB)",
    "peak_vram_mb":       "VRAM pico (MB)",
}

QUALITY_METRICS = {
    "bertscore_f1": "BERTScore F1",
    "rougeL_f1":    "ROUGE-L F1",
}

# ── Leitura ───────────────────────────────────────────────────
print(f"Lendo: {INPUT_CSV}\n")
df = pd.read_csv(INPUT_CSV)

if "success" in df.columns:
    df = df[df["success"].astype(str).str.lower() == "true"].copy()

if df["rag"].dtype != bool:
    df["rag"] = df["rag"].astype(str).str.lower().isin(["true", "1", "yes"])

df["machine_name"] = df["machine_name"].astype(str).str.strip()
df["quantization"] = df["quantization"].astype(str).str.strip()

# ── Teste ─────────────────────────────────────────────────────
def mannwhitney(a, b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - (2 * stat) / (len(a) * len(b))   # rank-biserial correlation
    return stat, p, r, len(a), len(b)

def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < ALPHA: return "*"
    return "ns"

rows = []

for machine in ["desktop", "notebook"]:
    sub = df[df["machine_name"] == machine]

    # 1. Quantizacao: 4-bit vs 8-bit (colapsando sobre RAG)
    for metric, label in PERFORMANCE_METRICS.items():
        a = sub[sub["quantization"] == "4-bit"][metric].values
        b = sub[sub["quantization"] == "8-bit"][metric].values
        U, p, r, na, nb = mannwhitney(a, b)
        rows.append({
            "comparacao": "Quantizacao (4-bit vs 8-bit)",
            "metrica":    label,
            "machine":    machine,
            "n_4bit":     na,
            "n_8bit":     nb,
            "U":          round(U, 1),
            "p_value":    round(p, 6),
            "sig":        sig(p),
            "effect_r":   round(r, 3),
            "media_a":    round(float(np.nanmean(a)), 4),
            "media_b":    round(float(np.nanmean(b)), 4),
        })

    # 2. RAG: com RAG vs sem RAG (colapsando sobre quantizacao)
    for metric, label in QUALITY_METRICS.items():
        a = sub[sub["rag"] == True][metric].values
        b = sub[sub["rag"] == False][metric].values
        U, p, r, na, nb = mannwhitney(a, b)
        rows.append({
            "comparacao": "RAG (com RAG vs sem RAG)",
            "metrica":    label,
            "machine":    machine,
            "n_4bit":     na,
            "n_8bit":     nb,
            "U":          round(U, 1),
            "p_value":    round(p, 6),
            "sig":        sig(p),
            "effect_r":   round(r, 3),
            "media_a":    round(float(np.nanmean(a)), 4),
            "media_b":    round(float(np.nanmean(b)), 4),
        })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_CSV, index=False)

# ── Impressao ─────────────────────────────────────────────────
for comp in result["comparacao"].unique():
    print(f"\n{'='*65}")
    print(f"  {comp}")
    print(f"{'='*65}")
    sub = result[result["comparacao"] == comp]
    print(sub[["metrica","machine","media_a","media_b","U","p_value","sig","effect_r"]]
          .to_string(index=False))

print("\nLegenda: *** p<0.001 | ** p<0.01 | * p<0.05 | ns = nao significativo")
print(f"\nSalvo: {OUTPUT_CSV}")

# ── Tabela LaTeX ──────────────────────────────────────────────
latex_rows = []
for comp, comp_label, metrics in [
    ("Quantizacao (4-bit vs 8-bit)", "Quantizacao", PERFORMANCE_METRICS),
    ("RAG (com RAG vs sem RAG)",     "RAG",         QUALITY_METRICS),
]:
    first = True
    for metric, label in metrics.items():
        sub = result[result["comparacao"] == comp]
        d = sub[(sub["metrica"] == label) & (sub["machine"] == "desktop")].iloc[0]
        n = sub[(sub["metrica"] == label) & (sub["machine"] == "notebook")].iloc[0]

        comp_col = f"\\textbf{{{comp_label}}}" if first else ""
        first = False

        latex_rows.append(
            f"  {comp_col} & {label}"
            f" & {d['media_a']:.3f} & {d['media_b']:.3f} & {d['p_value']:.4f} & {d['sig']} & {d['effect_r']:.3f}"
            f" & {n['media_a']:.3f} & {n['media_b']:.3f} & {n['p_value']:.4f} & {n['sig']} & {n['effect_r']:.3f}"
            f" \\\\"
        )
    latex_rows.append("  \\midrule")

latex_rows.pop()  # remove ultimo midrule

latex = r"""\begin{table}[ht]
\centering
\small
\caption{Resultados do teste de Mann-Whitney U para as comparacoes principais.
$\bar{x}_A$ e $\bar{x}_B$ sao as medias dos grupos comparados;
$r$ e o coeficiente de correlacao de postos (tamanho de efeito).
Significancia: *** $p{<}0{,}001$; ** $p{<}0{,}01$; * $p{<}0{,}05$; ns nao significativo.}
\label{tab:hypothesis}
\begin{tabular}{llrrrrrrrrrr}
\toprule
& & \multicolumn{5}{c}{\textbf{Desktop}} & \multicolumn{5}{c}{\textbf{Notebook}} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}
\textbf{Comparacao} & \textbf{Metrica}
  & $\bar{x}_A$ & $\bar{x}_B$ & $p$ & sig & $r$
  & $\bar{x}_A$ & $\bar{x}_B$ & $p$ & sig & $r$ \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write(latex)

print(f"Salvo: {OUTPUT_TEX}")
