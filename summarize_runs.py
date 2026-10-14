import pandas as pd
from pathlib import Path

# -----------------------------
# ROOT DIRECTORY
# -----------------------------
ROOT = Path("/home/locolinux2/U24_synthesis/experiments")

# -----------------------------
# RUN GROUPS
# -----------------------------
RUNS = {
    "baseline": {
        "convnext_tiny": [
            "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed42_Augsgeometric_real1.0_syn0.0",
            "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed43_Augsgeometric_real1.0_syn0.0",
            "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed44_Augsgeometric_real1.0_syn0.0",
            "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed45_Augsgeometric_real1.0_syn0.0",
            "156_EMBED_binary_256x256_holdout_convnext_tiny_Replacement_seed46_Augsgeometric_real1.0_syn0.0",
        ],
        "efficientnet_b0": [
            "161_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed42_Augsgeometric_real1.0_syn0.0",
            "161_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed43_Augsgeometric_real1.0_syn0.0",
            "161_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed44_Augsgeometric_real1.0_syn0.0",
            "161_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed45_Augsgeometric_real1.0_syn0.0",
            "161_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed46_Augsgeometric_real1.0_syn0.0",
        ],
        "resnet50": [
            "166_EMBED_binary_256x256_holdout_resnet50_mulseed_seed42_Augsgeometric_real1.0_syn0.0",
            "166_EMBED_binary_256x256_holdout_resnet50_mulseed_seed43_Augsgeometric_real1.0_syn0.0",
            "166_EMBED_binary_256x256_holdout_resnet50_mulseed_seed44_Augsgeometric_real1.0_syn0.0",
            "166_EMBED_binary_256x256_holdout_resnet50_mulseed_seed45_Augsgeometric_real1.0_syn0.0",
            "166_EMBED_binary_256x256_holdout_resnet50_mulseed_seed46_Augsgeometric_real1.0_syn0.0",
        ],
    },

    "syn025": {
        "phase1": {
            "convnext_tiny": [
                "157_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed42_Augsgeometric_real1.0_syn0.25",
                "157_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed43_Augsgeometric_real1.0_syn0.25",
                "157_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed44_Augsgeometric_real1.0_syn0.25",
                "157_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed45_Augsgeometric_real1.0_syn0.25",
                "157_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed46_Augsgeometric_real1.0_syn0.25",
            ],
            "efficientnet_b0": [
                "162_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed42_Augsgeometric_real1.0_syn0.25",
                "162_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed43_Augsgeometric_real1.0_syn0.25",
                "162_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed44_Augsgeometric_real1.0_syn0.25",
                "162_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed45_Augsgeometric_real1.0_syn0.25",
                "162_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed46_Augsgeometric_real1.0_syn0.25",
            ],
            "resnet50": [
                "167_EMBED_binary_256x256_holdout_resnet50_mulseed_seed42_Augsgeometric_real1.0_syn0.25",
                "167_EMBED_binary_256x256_holdout_resnet50_mulseed_seed43_Augsgeometric_real1.0_syn0.25",
                "167_EMBED_binary_256x256_holdout_resnet50_mulseed_seed44_Augsgeometric_real1.0_syn0.25",
                "167_EMBED_binary_256x256_holdout_resnet50_mulseed_seed45_Augsgeometric_real1.0_syn0.25",
                "167_EMBED_binary_256x256_holdout_resnet50_mulseed_seed46_Augsgeometric_real1.0_syn0.25",
            ],
        },
        "phase2": {
            "convnext_tiny": [
                "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
                "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
                "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
                "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
                "159_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_157_real1.0_syn0.25",
            ],
            "efficientnet_b0": [
                "164_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_162_real1.0_syn0.25",
                "164_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_162_real1.0_syn0.25",
                "164_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_162_real1.0_syn0.25",
                "164_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_162_real1.0_syn0.25",
                "164_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_162_real1.0_syn0.25",
            ],
            "resnet50": [
                "169_EMBED_binary_256x256_holdout_resnet50_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_167_real1.0_syn0.25",
                "169_EMBED_binary_256x256_holdout_resnet50_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_167_real1.0_syn0.25",
                "169_EMBED_binary_256x256_holdout_resnet50_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_167_real1.0_syn0.25",
                "169_EMBED_binary_256x256_holdout_resnet50_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_167_real1.0_syn0.25",
                "169_EMBED_binary_256x256_holdout_resnet50_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_167_real1.0_syn0.25",
            ],
        },
    },

    "syn075": {
        "phase1": {
            "convnext_tiny": [
                "158_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed42_Augsgeometric_real1.0_syn0.75",
                "158_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed43_Augsgeometric_real1.0_syn0.75",
                "158_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed44_Augsgeometric_real1.0_syn0.75",
                "158_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed45_Augsgeometric_real1.0_syn0.75",
                "158_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed46_Augsgeometric_real1.0_syn0.75",
            ],
            "efficientnet_b0": [
                "163_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed42_Augsgeometric_real1.0_syn0.75",
                "163_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed43_Augsgeometric_real1.0_syn0.75",
                "163_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed44_Augsgeometric_real1.0_syn0.75",
                "163_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed45_Augsgeometric_real1.0_syn0.75",
                "163_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed46_Augsgeometric_real1.0_syn0.75",
            ],
            "resnet50": [
                "168_EMBED_binary_256x256_holdout_resnet50_mulseed_seed42_Augsgeometric_real1.0_syn0.75",
                "168_EMBED_binary_256x256_holdout_resnet50_mulseed_seed43_Augsgeometric_real1.0_syn0.75",
                "168_EMBED_binary_256x256_holdout_resnet50_mulseed_seed44_Augsgeometric_real1.0_syn0.75",
                "168_EMBED_binary_256x256_holdout_resnet50_mulseed_seed45_Augsgeometric_real1.0_syn0.75",
                "168_EMBED_binary_256x256_holdout_resnet50_mulseed_seed46_Augsgeometric_real1.0_syn0.75",
            ],
        },
        "phase2": {
            "convnext_tiny": [
                "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
                "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
                "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
                "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
                "160_EMBED_binary_256x256_holdout_convnext_tiny_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_158_real1.0_syn0.75",
            ],
            "efficientnet_b0": [
                "165_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_163_real1.0_syn0.75",
                "165_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_163_real1.0_syn0.75",
                "165_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_163_real1.0_syn0.75",
                "165_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_163_real1.0_syn0.75",
                "165_EMBED_binary_256x256_holdout_efficientnet_b0_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_163_real1.0_syn0.75",
            ],
            "resnet50": [
                "170_EMBED_binary_256x256_holdout_resnet50_mulseed_seed42_Augsgeometric_real1.0_syn0.0_fTune_168_real1.0_syn0.75",
                "170_EMBED_binary_256x256_holdout_resnet50_mulseed_seed43_Augsgeometric_real1.0_syn0.0_fTune_168_real1.0_syn0.75",
                "170_EMBED_binary_256x256_holdout_resnet50_mulseed_seed44_Augsgeometric_real1.0_syn0.0_fTune_168_real1.0_syn0.75",
                "170_EMBED_binary_256x256_holdout_resnet50_mulseed_seed45_Augsgeometric_real1.0_syn0.0_fTune_168_real1.0_syn0.75",
                "170_EMBED_binary_256x256_holdout_resnet50_mulseed_seed46_Augsgeometric_real1.0_syn0.0_fTune_168_real1.0_syn0.75",
            ],
        },
    },
}

# -----------------------------
# METRIC EXTRACTION
# -----------------------------
def extract_best_metrics(run_dir: str) -> dict:
    logs_path = ROOT / run_dir / "logs.csv"
    if not logs_path.exists():
        raise FileNotFoundError(f"Missing logs.csv: {logs_path}")

    df = pd.read_csv(logs_path)

    required_cols = {"phase", "balanced_acc", "ACC_class0", "ACC_class1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{logs_path} is missing columns: {sorted(missing)}")

    val_df = df[df["phase"] == "val"].copy()
    if val_df.empty:
        raise ValueError(f"No validation rows found in {logs_path}")

    best = val_df.loc[val_df["balanced_acc"].idxmax()]

    return {
        "run_dir": run_dir,
        "balanced_acc": float(best["balanced_acc"]),
        "specificity": float(best["ACC_class0"]),
        "sensitivity": float(best["ACC_class1"]),
    }

# -----------------------------
# PROCESS A GROUP
# -----------------------------
def process_group(dirs, condition, phase, model):
    rows = []
    for d in dirs:
        metrics = extract_best_metrics(d)
        metrics["condition"] = condition
        metrics["phase"] = phase
        metrics["model"] = model
        rows.append(metrics)
    return pd.DataFrame(rows)

# -----------------------------
# COLLECT ALL RESULTS
# -----------------------------
all_parts = []

for condition, spec in RUNS.items():
    if condition == "baseline":
        for model, dirs in spec.items():
            all_parts.append(process_group(dirs, condition, "baseline", model))
    else:
        for phase, model_map in spec.items():
            for model, dirs in model_map.items():
                all_parts.append(process_group(dirs, condition, phase, model))

runs_df = pd.concat(all_parts, ignore_index=True)

# -----------------------------
# SUMMARY
# -----------------------------
summary_df = (
    runs_df
    .groupby(["condition", "phase", "model"], as_index=False)
    .agg(
        balanced_acc_mean=("balanced_acc", "mean"),
        balanced_acc_std=("balanced_acc", "std"),
        sensitivity_mean=("sensitivity", "mean"),
        sensitivity_std=("sensitivity", "std"),
        specificity_mean=("specificity", "mean"),
        specificity_std=("specificity", "std"),
        n=("balanced_acc", "size"),
    )
)

condition_order = ["baseline", "syn025", "syn075"]
phase_order = ["baseline", "phase1", "phase2"]
model_order = ["convnext_tiny", "efficientnet_b0", "resnet50"]

summary_df["condition"] = pd.Categorical(summary_df["condition"], categories=condition_order, ordered=True)
summary_df["phase"] = pd.Categorical(summary_df["phase"], categories=phase_order, ordered=True)
summary_df["model"] = pd.Categorical(summary_df["model"], categories=model_order, ordered=True)

summary_df = summary_df.sort_values(["model", "condition", "phase"]).reset_index(drop=True)
runs_df["condition"] = pd.Categorical(runs_df["condition"], categories=condition_order, ordered=True)
runs_df["phase"] = pd.Categorical(runs_df["phase"], categories=phase_order, ordered=True)
runs_df["model"] = pd.Categorical(runs_df["model"], categories=model_order, ordered=True)
runs_df = runs_df.sort_values(["model", "condition", "phase", "run_dir"]).reset_index(drop=True)

# -----------------------------
# PRINT RESULTS
# -----------------------------
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

print("\n=== PER-RUN METRICS ===\n")
print(runs_df.to_string(index=False))

print("\n=== SUMMARY (mean ± std) ===\n")
for model in model_order:
    print(f"{model}:")
    block = summary_df[summary_df["model"] == model]
    for _, row in block.iterrows():
        label = f"{row['condition']}" if row["condition"] == "baseline" else f"{row['condition']} / {row['phase']}"
        print(f"  {label}:")
        print(f"    Balanced Acc: {row['balanced_acc_mean']:.4f} ± {row['balanced_acc_std']:.4f}")
        print(f"    Sensitivity : {row['sensitivity_mean']:.4f} ± {row['sensitivity_std']:.4f}")
        print(f"    Specificity : {row['specificity_mean']:.4f} ± {row['specificity_std']:.4f}")
    print()

# -----------------------------
# SAVE CSVs
# -----------------------------
runs_df.to_csv("all_run_metrics.csv", index=False)
summary_df.to_csv("summary_results_by_model_and_phase.csv", index=False)

print("\nSaved:")
print("  all_run_metrics.csv")
print("  summary_results_by_model_and_phase.csv")