from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


# =========================
# Caminhos
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

INPUT_CSV = PROJECT_DIR / "results" / "results_combined.csv"

OUTPUT_SUMMARY_CSV = PROJECT_DIR / "results" / "summary_with_ci95.csv"
OUTPUT_TABLE_CSV = PROJECT_DIR / "results" / "table_results_ci95.csv"
OUTPUT_LATEX_TABLE = PROJECT_DIR / "results" / "table_results_ci95.tex"


# =========================
# Configurações
# =========================

GROUP_COLS = ["machine_name", "quantization", "rag"]

METRICS = {
    "latency_s": {
        "label": "Lat. (s)",
        "decimals": 2
    },
    "throughput_tokens_s": {
        "label": "Thr.",
        "decimals": 2
    },
    "bertscore_f1": {
        "label": "BERTScore",
        "decimals": 3
    },
    "rougeL_f1": {
        "label": "ROUGE-L",
        "decimals": 3
    }
}


# =========================
# Funções auxiliares
# =========================

def ci95_margin(std, n):
    if n <= 1 or pd.isna(std):
        return np.nan

    se = std / np.sqrt(n)
    return stats.t.ppf(0.975, df=n - 1) * se


def format_mean_ci(mean, margin, decimals):
    if pd.isna(mean) or pd.isna(margin):
        return "--"

    return f"{mean:.{decimals}f} $\\pm$ {margin:.{decimals}f}"


def format_config(row):
    q = str(row["quantization"])

    if str(row["rag"]).lower() in ["true", "1", "yes"]:
        rag = "com RAG"
    else:
        rag = "sem RAG"

    return f"{q} {rag}"


def format_machine(machine):
    machine_str = str(machine).lower()

    if machine_str == "desktop":
        return "Desktop"

    if machine_str == "notebook":
        return "Notebook"

    return str(machine)


# =========================
# Leitura dos dados
# =========================

print(f"Lendo arquivo: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

print("Colunas encontradas no CSV:")
print(list(df.columns))
print()

# Mantém apenas execuções bem-sucedidas, se a coluna existir
if "success" in df.columns:
    df = df[df["success"] == True].copy()

# Garante que rag é booleano
if "rag" in df.columns:
    if df["rag"].dtype != bool:
        df["rag"] = df["rag"].astype(str).str.lower().isin(["true", "1", "yes"])


# Verifica se as colunas necessárias existem
required_cols = set(GROUP_COLS + list(METRICS.keys()))
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(
        "As seguintes colunas esperadas não foram encontradas no CSV: "
        + ", ".join(missing_cols)
    )


# =========================
# Cálculo dos ICs
# =========================

summary_rows = []

for metric in METRICS.keys():
    grouped = (
        df
        .groupby(GROUP_COLS)[metric]
        .agg(
            n="count",
            mean="mean",
            std="std"
        )
        .reset_index()
    )

    grouped["se"] = grouped["std"] / np.sqrt(grouped["n"])
    grouped["ci95_margin"] = grouped.apply(
        lambda row: ci95_margin(row["std"], row["n"]),
        axis=1
    )
    grouped["ci95_lower"] = grouped["mean"] - grouped["ci95_margin"]
    grouped["ci95_upper"] = grouped["mean"] + grouped["ci95_margin"]
    grouped["metric"] = metric

    summary_rows.append(grouped)

summary_long = pd.concat(summary_rows, ignore_index=True)

summary_long = summary_long[
    GROUP_COLS + [
        "metric",
        "n",
        "mean",
        "std",
        "se",
        "ci95_lower",
        "ci95_upper",
        "ci95_margin"
    ]
]

summary_long.to_csv(OUTPUT_SUMMARY_CSV, index=False)


# =========================
# Tabela em formato largo
# =========================

table_rows = []

configs = (
    df[GROUP_COLS]
    .drop_duplicates()
    .sort_values(["machine_name", "quantization", "rag"])
)

for _, config_row in configs.iterrows():
    row = {
        "Máquina": format_machine(config_row["machine_name"]),
        "Configuração": format_config(config_row)
    }

    subset = summary_long[
        (summary_long["machine_name"] == config_row["machine_name"]) &
        (summary_long["quantization"] == config_row["quantization"]) &
        (summary_long["rag"] == config_row["rag"])
    ]

    n_values = subset["n"].dropna().unique()
    row["n"] = int(n_values[0]) if len(n_values) > 0 else None

    for metric, info in METRICS.items():
        metric_row = subset[subset["metric"] == metric].iloc[0]

        row[info["label"]] = format_mean_ci(
            metric_row["mean"],
            metric_row["ci95_margin"],
            info["decimals"]
        )

    table_rows.append(row)

table_df = pd.DataFrame(table_rows)

table_df.to_csv(OUTPUT_TABLE_CSV, index=False)


# =========================
# Geração da tabela LaTeX
# =========================

latex = table_df.to_latex(
    index=False,
    escape=False,
    column_format="llrrrrr"
)

caption = (
    "\\caption{Resultados agregados por máquina, quantização e uso de RAG. "
    "Os valores são apresentados como média $\\pm$ intervalo de confiança de 95\\%, "
    "considerando as execuções bem-sucedidas de cada configuração experimental.}"
)

label = "\\label{tab:results-ci95}"

latex = latex.replace(
    "\\begin{tabular}",
    "\\begin{table}[ht]\n\\centering\n\\small\n"
    + caption
    + "\n"
    + label
    + "\n\\begin{tabular}"
)

latex = latex.replace(
    "\\end{tabular}",
    "\\end{tabular}\n\\end{table}"
)

with open(OUTPUT_LATEX_TABLE, "w", encoding="utf-8") as f:
    f.write(latex)


print("Arquivos gerados:")
print(f"- {OUTPUT_SUMMARY_CSV}")
print(f"- {OUTPUT_TABLE_CSV}")
print(f"- {OUTPUT_LATEX_TABLE}")
print()
print(table_df)