import pandas as pd

# ==================================================
# LOAD DESKTOP CSV
# ==================================================

desktop = pd.read_csv(
    "../results/results_with_metrics.csv"
)

desktop["machine_name"] = "desktop"

# ==================================================
# LOAD NOTEBOOK CSV
# ==================================================

notebook = pd.read_csv(
    "../results/results_notebook_with_metrics.csv"
)

notebook["machine_name"] = "notebook"

# ==================================================
# CONCAT
# ==================================================

combined = pd.concat(
    [
        desktop,
        notebook
    ],
    ignore_index=True
)

# ==================================================
# SAVE
# ==================================================

combined.to_csv(
    "../results/results_combined.csv",
    index=False
)

# ==================================================
# CHECK
# ==================================================

print("\n===================================")
print("ROWS")
print("===================================")

print(len(combined))

print("\n===================================")
print("MACHINES")
print("===================================")

print(
    combined["machine_name"]
    .value_counts()
)

print("\n===================================")
print("CONFIGS")
print("===================================")

print(
    combined.groupby(
        [
            "machine_name",
            "quantization",
            "rag"
        ]
    )
    .size()
)

print("\n===================================")
print("SAVED")
print("===================================")

print(
    "../results/results_combined.csv"
)