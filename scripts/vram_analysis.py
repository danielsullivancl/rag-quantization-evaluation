import pandas as pd

from config import OUTPUT_CSV

df = pd.read_csv(OUTPUT_CSV)

print("\n===================================")
print("BASELINE VRAM")
print("===================================")

print(
    df.groupby(
        ["quantization", "rag"]
    )[
        [
            "baseline_vram_mb",
            "peak_vram_mb",
            "vram_delta_mb"
        ]
    ]
    .mean()
)

print("\n===================================")
print("TOP BASELINE VRAM")
print("===================================")

print(
    df[
        [
            "quantization",
            "rag",
            "baseline_vram_mb",
            "peak_vram_mb",
            "vram_delta_mb"
        ]
    ]
    .sort_values(
        "baseline_vram_mb",
        ascending=False
    )
    .head(20)
)