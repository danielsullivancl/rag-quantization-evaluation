import pandas as pd
import numpy as np

from config import OUTPUT_CSV

# ==================================================
# LOAD CSV
# ==================================================

print("Carregando CSV...")

df = pd.read_csv(OUTPUT_CSV)

# ==================================================
# INFO GERAL
# ==================================================

print("\n===================================")
print("DATASET INFO")
print("===================================")

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ==================================================
# EXPECTED ROWS
# ==================================================

EXPECTED_ROWS = 450 * 2 * 2

print("\n===================================")
print("EXPECTED ROWS")
print("===================================")

print(f"Expected: {EXPECTED_ROWS}")
print(f"Actual: {len(df)}")

if len(df) == EXPECTED_ROWS:
    print("OK")
else:
    print("WARNING: Missing rows")

# ==================================================
# DUPLICATES
# ==================================================

duplicates = df.duplicated(
    subset=[
        "sample_index",
        "quantization",
        "rag"
    ]
).sum()

print("\n===================================")
print("DUPLICATES")
print("===================================")

print(duplicates)

# ==================================================
# SUCCESS / FAILURE
# ==================================================

print("\n===================================")
print("SUCCESS / FAILURE")
print("===================================")

print(
    df["success"].value_counts(dropna=False)
)

# ==================================================
# FAILED EXECUTIONS
# ==================================================

failed = df[df["success"] == False]

print("\n===================================")
print("FAILED EXECUTIONS")
print("===================================")

print(len(failed))

if len(failed) > 0:

    print("\nMost common errors:\n")

    print(
        failed["error"]
        .value_counts()
        .head(10)
    )

# ==================================================
# NaN CHECK
# ==================================================

print("\n===================================")
print("NaN COUNT")
print("===================================")

nan_counts = df.isna().sum()

print(
    nan_counts[
        nan_counts > 0
    ]
)

# ==================================================
# CONFIG COUNTS
# ==================================================

print("\n===================================")
print("EXECUTIONS PER CONFIG")
print("===================================")

print(
    df.groupby(
        ["quantization", "rag"]
    ).size()
)

# ==================================================
# UNIQUE QUESTIONS
# ==================================================

print("\n===================================")
print("UNIQUE SAMPLE INDEX")
print("===================================")

print(
    df["sample_index"].nunique()
)

# ==================================================
# NUMERIC COLUMNS
# ==================================================

numeric_cols = [
    "retrieval_time_s",
    "prompt_build_time_s",
    "ollama_latency_s",
    "end_to_end_latency_s",
    "throughput_tokens_s",
    "prompt_token_count",
    "output_token_count",
    "ram_usage_mb",
    "vram_delta_mb"
]

# ==================================================
# DESCRIPTIVE STATS
# ==================================================

print("\n===================================")
print("DESCRIPTIVE STATS")
print("===================================")

print(
    df[numeric_cols]
    .describe()
)

# ==================================================
# NEGATIVE VALUES
# ==================================================

print("\n===================================")
print("NEGATIVE VALUES")
print("===================================")

for col in numeric_cols:

    negative_count = (
        df[col] < 0
    ).sum()

    print(f"{col}: {negative_count}")

# ==================================================
# INFINITE VALUES
# ==================================================

print("\n===================================")
print("INFINITE VALUES")
print("===================================")

for col in numeric_cols:

    inf_count = np.isinf(
        df[col]
    ).sum()

    print(f"{col}: {inf_count}")

# ==================================================
# TOP LATENCIES
# ==================================================

print("\n===================================")
print("TOP 10 END-TO-END LATENCIES")
print("===================================")

top_latency = (
    df.sort_values(
        "end_to_end_latency_s",
        ascending=False
    )
    [
        [
            "sample_index",
            "quantization",
            "rag",
            "end_to_end_latency_s",
            "question"
        ]
    ]
    .head(10)
)

print(top_latency)

# ==================================================
# QUICK COMPARISON
# ==================================================

print("\n===================================")
print("MEAN LATENCIES")
print("===================================")

comparison = (
    df.groupby(
        ["quantization", "rag"]
    )[
        [
            "retrieval_time_s",
            "ollama_latency_s",
            "end_to_end_latency_s"
        ]
    ]
    .mean()
)

print(comparison)

# ==================================================
# MEMORY COMPARISON
# ==================================================

print("\n===================================")
print("MEAN MEMORY USAGE")
print("===================================")

memory_comparison = (
    df.groupby(
        ["quantization", "rag"]
    )[
        [
            "ram_usage_mb",
            "vram_delta_mb"
        ]
    ]
    .mean()
)

print(memory_comparison)

# ==================================================
# THROUGHPUT
# ==================================================

print("\n===================================")
print("MEAN THROUGHPUT")
print("===================================")

throughput_comparison = (
    df.groupby(
        ["quantization", "rag"]
    )[
        "throughput_tokens_s"
    ]
    .mean()
)

print(throughput_comparison)

# ==================================================
# SAVE CLEAN COPY
# ==================================================

clean_df = df[
    df["success"] == True
].copy()

clean_output = OUTPUT_CSV.replace(
    ".csv",
    "_clean.csv"
)

clean_df.to_csv(
    clean_output,
    index=False
)

print("\n===================================")
print("CLEAN CSV SAVED")
print("===================================")

print(clean_output)

# ==================================================
# FINISHED
# ==================================================

print("\n===================================")
print("SANITY CHECK FINISHED")
print("===================================")