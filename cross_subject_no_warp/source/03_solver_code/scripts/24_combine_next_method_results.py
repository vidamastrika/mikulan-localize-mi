from pathlib import Path
import pandas as pd

OUTPUTS = Path("outputs")
DEST = OUTPUTS / "23_full_next_method_grid"

ROOTS = [
    "23_full_next_method_grid",
    "23_irmxne_alpha20_extension",
    "23_irmxne_alpha15_extension",
    "23_champagne_pruning_extension",
    "23_gamma_xyz_true",
    "23_gamma_depth_test",
    "23_gamma_alpha_extension",
    "23_gamma_alpha_low_extension",
    "23_sesame_fs_extension",
]

METHODS = {"irmxne", "champagne", "gamma_map", "sesame"}

frames = []
summary = []
seen = set()

for root_name in ROOTS:
    root = OUTPUTS / root_name

    if not root.exists():
        print(f"Missing folder: {root}")
        continue

    for path in sorted(root.rglob("grid_results.csv")):
        resolved = path.resolve()

        if resolved in seen:
            continue
        seen.add(resolved)

        df = pd.read_csv(path, low_memory=False)

        if df.empty:
            continue

        if "method" not in df.columns:
            print(f"SKIPPED (no method column): {path}")
            continue

        df["method_original"] = df["method"]
        df["method"] = (
            df["method"].astype(str).str.strip().str.lower()
            .str.replace("-", "_", regex=False)
        )

        df = df[df["method"].isin(METHODS)].copy()

        if df.empty:
            continue

        # Keep every original experimental result.
        df["source_file"] = str(path)
        df["source_experiment"] = root_name
        df["source_row"] = df.index + 2

        frames.append(df)

        for method, group in df.groupby("method"):
            statuses = (
                group["status"].astype(str).str.upper()
                if "status" in group.columns
                else pd.Series(dtype=str)
            )

            summary.append({
                "source_experiment": root_name,
                "source_file": str(path),
                "method": method,
                "rows": len(group),
                "pass": (statuses == "PASS").sum(),
                "fail": (statuses == "FAIL").sum(),
            })

        print(f"READ: {path} ({len(df):,} rows)")

if not frames:
    raise SystemExit("No suitable result CSV files found.")

combined = pd.concat(frames, ignore_index=True, sort=False)

# Reorganize columns, preserving all original fields.
first = [
    "method",
    "subject",
    "run",
    "configuration_id",
    "montage",
    "status",
]

last = ["source_experiment", "source_file", "source_row"]

middle = [
    c for c in combined.columns
    if c not in first + last
]

order = (
    [c for c in first if c in combined.columns]
    + middle
    + last
)

combined = combined[order]

DEST.mkdir(parents=True, exist_ok=True)

master_path = DEST / "23_all_next_methods_combined.csv"
summary_path = DEST / "23_combined_source_summary.csv"

combined.to_csv(master_path, index=False)
pd.DataFrame(summary).to_csv(summary_path, index=False)

print("\n=== COMBINED RESULTS ===")

for method, group in combined.groupby("method"):
    print(f"{method:12s}: {len(group):,} rows")

print(f"\nTOTAL: {len(combined):,} rows")
print(f"Master CSV: {master_path}")
print(f"Source summary: {summary_path}")
print("\nOriginal CSV files were not modified.")
