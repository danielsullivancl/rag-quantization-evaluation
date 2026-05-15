import pandas as pd
import numpy as np

from bert_score import score as bertscore
from rouge_score import rouge_scorer

from config import OUTPUT_CSV

# ==================================================
# LOAD CSV
# ==================================================

print("Carregando CSV...")

df = pd.read_csv(OUTPUT_CSV)

print(f"Rows: {len(df)}")

# ==================================================
# CLEAN TEXT
# ==================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    return str(text).strip()

df["expected"] = df["expected"].apply(clean_text)
df["response"] = df["response"].apply(clean_text)

# ==================================================
# BERTSCORE
# ==================================================

print("\n===================================")
print("CALCULATING BERTSCORE")
print("===================================")

predictions = df["response"].tolist()
references = df["expected"].tolist()

P, R, F1 = bertscore(
    predictions,
    references,
    lang="en",
    verbose=True
)

df["bertscore_precision"] = P.numpy()
df["bertscore_recall"] = R.numpy()
df["bertscore_f1"] = F1.numpy()

# ==================================================
# ROUGE
# ==================================================

print("\n===================================")
print("CALCULATING ROUGE")
print("===================================")

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

rouge1_list = []
rouge2_list = []
rougeL_list = []

for _, row in df.iterrows():

    scores = scorer.score(
        row["expected"],
        row["response"]
    )

    rouge1_list.append(
        scores["rouge1"].fmeasure
    )

    rouge2_list.append(
        scores["rouge2"].fmeasure
    )

    rougeL_list.append(
        scores["rougeL"].fmeasure
    )

df["rouge1_f1"] = rouge1_list
df["rouge2_f1"] = rouge2_list
df["rougeL_f1"] = rougeL_list

# ==================================================
# EXACT MATCH
# ==================================================

print("\n===================================")
print("CALCULATING EXACT MATCH")
print("===================================")

def exact_match(a, b):

    return int(
        clean_text(a).lower()
        ==
        clean_text(b).lower()
    )

df["exact_match"] = df.apply(
    lambda row: exact_match(
        row["expected"],
        row["response"]
    ),
    axis=1
)

# ==================================================
# SAVE NEW CSV
# ==================================================

output_metrics_csv = OUTPUT_CSV.replace(
    ".csv",
    "_with_metrics.csv"
)

df.to_csv(
    output_metrics_csv,
    index=False
)

print("\n===================================")
print("CSV SAVED")
print("===================================")

print(output_metrics_csv)

# ==================================================
# AGGREGATED RESULTS
# ==================================================

print("\n===================================")
print("AGGREGATED RESULTS")
print("===================================")

summary = (
    df.groupby(
        ["quantization", "rag"]
    )[
        [
            "bertscore_f1",
            "rougeL_f1",
            "exact_match",
            "end_to_end_latency_s",
            "throughput_tokens_s",
            "baseline_vram_mb",
            "ram_usage_mb"
        ]
    ]
    .mean()
)

print(summary)

# ==================================================
# BEST / WORST
# ==================================================

print("\n===================================")
print("BEST BERTSCORE")
print("===================================")

best = df.sort_values(
    "bertscore_f1",
    ascending=False
)[
    [
        "quantization",
        "rag",
        "bertscore_f1",
        "question"
    ]
].head(10)

print(best)

print("\n===================================")
print("WORST BERTSCORE")
print("===================================")

worst = df.sort_values(
    "bertscore_f1",
    ascending=True
)[
    [
        "quantization",
        "rag",
        "bertscore_f1",
        "question"
    ]
].head(10)

print(worst)

print("\n===================================")
print("FINISHED")
print("===================================")