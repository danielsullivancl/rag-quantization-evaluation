import pandas as pd

from config import OUTPUT_CSV

# ==================================================
# LOAD CSV
# ==================================================

print("Carregando resultados...")

df = pd.read_csv(OUTPUT_CSV)

# ==================================================
# INFO GERAL
# ==================================================

print("\n===================================")
print("TOTAL ROWS")
print("===================================")

print(len(df))

# ==================================================
# TOTAL ESPERADO
# ==================================================

# 450 perguntas
# 2 quantizações
# 2 modos (RAG / sem RAG)

expected_rows = 450 * 2 * 2

print("\n===================================")
print("EXPECTED ROWS")
print("===================================")

print(expected_rows)

# ==================================================
# COMPLETO?
# ==================================================

print("\n===================================")
print("EXPERIMENT COMPLETE?")
print("===================================")

if len(df) == expected_rows:

    print("YES - ALL EXECUTIONS COMPLETED")

else:

    missing = expected_rows - len(df)

    print("NO")
    print(f"Missing rows: {missing}")

# ==================================================
# SUCCESS / FAILURE
# ==================================================

print("\n===================================")
print("SUCCESS / FAILURE")
print("===================================")

print(df["success"].value_counts())

# ==================================================
# FAILURES
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
# QUANTIZATION COUNTS
# ==================================================

print("\n===================================")
print("QUANTIZATION COUNTS")
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
print("UNIQUE QUESTIONS")
print("===================================")

print(df["id"].nunique())

# ==================================================
# DUPLICATES
# ==================================================

duplicates = df.duplicated(
    subset=["id", "quantization", "rag"]
).sum()

print("\n===================================")
print("DUPLICATES")
print("===================================")

print(duplicates)