import pandas as pd
from pathlib import Path

base_dir = Path("/home/locolinux2/U24_synthesis/experiments")

# Change this to 175 or 176
experiment_id = "175"

pattern = (
    f"{experiment_id}_inbreast_clean_256x256_holdout_convnext_tiny_"
    f"mulseed_seed*_Augsgeometric_real1.0_syn0.0*"
)

rows = []

for exp_dir in sorted(base_dir.glob(pattern)):
    log_path = exp_dir / "logs.csv"

    if not log_path.exists():
        print(f"Missing logs.csv: {exp_dir}")
        continue

    df = pd.read_csv(log_path)

    # epoch 1 training row
    row = df[(df["epoch"] == 1) & (df["phase"] == "train")].iloc[0]

    rows.append({
        "experiment": exp_dir.name,
        "balanced_acc": row["balanced_acc"],
        "sensitivity": row["ACC_class1"],   # positive class accuracy
        "specificity": row["ACC_class0"],   # negative class accuracy
    })

results = pd.DataFrame(rows)

print(f"\nFound {len(results)} experiments")

print("\nPer-experiment values:")
print(results.to_string(index=False))

metrics = ["balanced_acc", "sensitivity", "specificity"]

summary = pd.DataFrame({
    "mean": results[metrics].mean(),
    "std": results[metrics].std()
})

print("\nSummary statistics:")
print(summary)

# import pandas as pd
# from pathlib import Path

# base_dir = Path("/home/locolinux2/U24_synthesis/experiments")

# rows = []

# for exp_dir in base_dir.glob(
#     "176_inbreast_clean_256x256_holdout_convnext_tiny_mulseed_seed*_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75"
# ):
#     log_path = exp_dir / "logs.csv"

#     df = pd.read_csv(log_path)

#     # epoch 1 training row
#     row = df[(df["epoch"] == 1) & (df["phase"] == "train")].iloc[0]

#     rows.append({
#         "experiment": exp_dir.name,
#         "balanced_acc": row["balanced_acc"],
#         "sensitivity": row["ACC_class1"],
#         "specificity": row["ACC_class0"],
#     })

# results = pd.DataFrame(rows)

# print(f"\nFound {len(results)} experiments")

# print("\nPer-experiment values:")
# print(results.to_string(index=False))

# metrics = ["balanced_acc", "sensitivity", "specificity"]

# summary = pd.DataFrame({
#     "mean": results[metrics].mean(),
#     "std": results[metrics].std()
# })

# print("\nSummary statistics:")
# print(summary)