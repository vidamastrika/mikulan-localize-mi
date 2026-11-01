#!/usr/bin/env python3
"""
Script 09 — Visualize Results (PNG only)

Purpose
-------
Create publication-ready visualizations for the finalized baseline, LOSO, and
single-reference experiments. This revised version uses ordinary VERTICAL
BOXPLOTS for the main comparison and separates the previous combined master
figure into two independent PNG figures so each can stand alone on a paper page.

Key design choices
------------------
- Main comparison uses run-level data from Script 07 (01_master_run_level.csv).
- Figure 1: within-subject comparison only (Historical baseline vs LOSO within).
- Figure 2: cross-subject no-warp comparison only (LOSO and single-reference,
  primary and sensitivity).
- Method groups are shown as horizontal bands and top labels (not stacked at the
  left margin), to keep the composition clean.
- Method-year labels are preserved on the x-axis.
- PNG output only (no PDF).
- No coordinate adjustments, no warp, no alignment.

Inputs
------
Required
- cross_subject_no_warp/outputs/07_compare_strategies/01_master_run_level.csv
- cross_subject_no_warp/outputs/08_reference_sensitivity/02_reference_method_summary.csv
- cross_subject_no_warp/outputs/08_reference_sensitivity/04_method_sensitivity_summary.csv
- cross_subject_no_warp/outputs/08_reference_sensitivity/05_reference_pairing_matrix.csv

Outputs
-------
Directory:
- cross_subject_no_warp/outputs/09_visualize_results/

Files
-----
01_within_boxplot.png
02_cross_boxplot.png
03_pass_rate_grouped.png
04_reference_sensitivity.png
05_reference_by_method_heatmap.png
06_test_reference_heatmap.png
07_plot_manifest.json
README.txt

Example terminal commands
-------------------------
python cross_subject_no_warp/scripts/09_visualize_results.py --phase audit
python cross_subject_no_warp/scripts/09_visualize_results.py
python cross_subject_no_warp/scripts/09_visualize_results.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("cross_subject_no_warp")
DEFAULT_MASTER_RUN = PROJECT_ROOT / "outputs" / "07_compare_strategies" / "01_master_run_level.csv"
DEFAULT_REF_METHOD = PROJECT_ROOT / "outputs" / "08_reference_sensitivity" / "02_reference_method_summary.csv"
DEFAULT_METHOD_SENS = PROJECT_ROOT / "outputs" / "08_reference_sensitivity" / "04_method_sensitivity_summary.csv"
DEFAULT_PAIRING = PROJECT_ROOT / "outputs" / "08_reference_sensitivity" / "05_reference_pairing_matrix.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "09_visualize_results"

METHOD_META: Dict[str, Dict[str, str]] = {
    "MNE": {"group": "Minimum-norm", "year": "1994"},
    "dSPM": {"group": "Minimum-norm", "year": "2002"},
    "sLORETA": {"group": "Minimum-norm", "year": "2002"},
    "eLORETA": {"group": "Minimum-norm", "year": "2011"},
    "Continuous-ECD": {"group": "Dipole / Beamformer", "year": "1990"},
    "RAP-MUSIC": {"group": "Dipole / Beamformer", "year": "2001"},
    "LCMV": {"group": "Dipole / Beamformer", "year": "1997"},
    "MxNE": {"group": "Sparse", "year": "2011"},
    "irMxNE": {"group": "Sparse", "year": "2018"},
    "Gamma-MAP": {"group": "Bayesian", "year": "2008"},
    "Champagne": {"group": "Bayesian", "year": "2016"},
    "SESAME": {"group": "SESAME", "year": "2020"},
}

GROUP_ORDER = ["Minimum-norm", "Dipole / Beamformer", "Sparse", "Bayesian", "SESAME"]
METHOD_ORDER = [
    "MNE", "dSPM", "sLORETA", "eLORETA",
    "Continuous-ECD", "RAP-MUSIC", "LCMV",
    "MxNE", "irMxNE",
    "Gamma-MAP", "Champagne",
    "SESAME",
]

COLORS = {
    "Baseline (historical all-pass)": "#4E79A7",
    "LOSO within": "#F28E2B",
    "LOSO cross primary": "#59A14F",
    "LOSO cross sensitivity": "#8CD17D",
    "Single-reference primary": "#E15759",
    "Single-reference sensitivity": "#FF9D9A",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create publication-ready PNG visualizations.")
    p.add_argument("--phase", choices=["audit", "plot"], default="plot")
    p.add_argument("--master-run", type=Path, default=DEFAULT_MASTER_RUN)
    p.add_argument("--reference-method", type=Path, default=DEFAULT_REF_METHOD)
    p.add_argument("--method-sensitivity", type=Path, default=DEFAULT_METHOD_SENS)
    p.add_argument("--pairing", type=Path, default=DEFAULT_PAIRING)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def validate_inputs(paths: List[Path]) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already contains files: {path}. Use --overwrite to replace Script 09 outputs.")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.parent / f"{path.name}_backup_{stamp}"
        shutil.move(str(path), str(backup))
    path.mkdir(parents=True, exist_ok=True)


def load_data(args: argparse.Namespace):
    master = pd.read_csv(args.master_run)
    ref_method = pd.read_csv(args.reference_method)
    method_sens = pd.read_csv(args.method_sensitivity)
    pairing = pd.read_csv(args.pairing)
    return master, ref_method, method_sens, pairing


def method_label(method: str) -> str:
    year = METHOD_META.get(method, {}).get("year", "")
    return f"{method}\n({year})" if year else method


def group_ranges() -> Dict[str, Tuple[int, int]]:
    ranges = {}
    start = 0
    for group in GROUP_ORDER:
        methods = [m for m in METHOD_ORDER if METHOD_META[m]["group"] == group]
        end = start + len(methods) - 1
        ranges[group] = (start, end)
        start = end + 1
    return ranges


def add_group_spans_and_labels(ax: plt.Axes) -> None:
    ranges = group_ranges()
    shade = True
    for group in GROUP_ORDER:
        start, end = ranges[group]
        if shade:
            ax.axvspan(start + 0.5, end + 1.5, color="#f7f7f7", zorder=0)
        shade = not shade
        xmid = ((start + 1) + (end + 1)) / 2.0
        ax.text(
            xmid,
            1.03,
            group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def style_boxplot(bp, facecolor: str) -> None:
    for box in bp["boxes"]:
        box.set(facecolor=facecolor, edgecolor=facecolor, alpha=0.75)
    for whisker in bp["whiskers"]:
        whisker.set(color=facecolor, linewidth=1.5)
    for cap in bp["caps"]:
        cap.set(color=facecolor, linewidth=1.5)
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.8)
    for flier in bp.get("fliers", []):
        flier.set(marker="o", markerfacecolor=facecolor, markeredgecolor=facecolor, alpha=0.35, markersize=3)


def prepare_within_box_data(master: pd.DataFrame):
    df = master.copy()
    df = df[(df["status"] == "PASS") & pd.to_numeric(df["value_mm"], errors="coerce").notna()].copy()
    df["value_mm"] = pd.to_numeric(df["value_mm"], errors="coerce")
    baseline = df[(df["experiment"] == "Historical_within") & (df["scenario"] == "all_pass_selected")]
    loso = df[(df["experiment"] == "LOSO_within") & (df["scenario"] == "primary")]
    return baseline, loso


def prepare_cross_box_data(master: pd.DataFrame):
    df = master.copy()
    df = df[(df["status"] == "PASS") & pd.to_numeric(df["value_mm"], errors="coerce").notna()].copy()
    df["value_mm"] = pd.to_numeric(df["value_mm"], errors="coerce")
    loso_primary = df[(df["experiment"] == "LOSO_single_reference_no_warp") & (df["scenario"] == "primary")]
    loso_sens = df[(df["experiment"] == "LOSO_single_reference_no_warp") & (df["scenario"] == "sensitivity")]
    sr_primary = df[(df["experiment"] == "Single-reference_single_reference_no_warp") & (df["scenario"] == "primary")]
    sr_sens = df[(df["experiment"] == "Single-reference_single_reference_no_warp") & (df["scenario"] == "sensitivity")]
    return loso_primary, loso_sens, sr_primary, sr_sens


def box_values_by_method(df: pd.DataFrame) -> List[np.ndarray]:
    out = []
    for method in METHOD_ORDER:
        vals = df.loc[df["method"] == method, "value_mm"].dropna().to_numpy(float)
        out.append(vals)
    return out


def create_master_three_strategy_boxplot(master: pd.DataFrame, out: Path) -> None:
    """Single-page grouped vertical boxplots, three strategies per method.

    Each method has its own color. Within each method, hatch and opacity
    distinguish Baseline, LOSO and Single-reference. The scenarios have
    different metric interpretations; axis annotation makes this explicit.
    """
    baseline, _ = prepare_within_box_data(master)
    loso, _, single, _ = prepare_cross_box_data(master)
    series = [
        ("Baseline", baseline, -0.25, 0.95, ""),
        ("LOSO", loso, 0.00, 0.65, "//"),
        ("Single-reference", single, +0.25, 0.40, "xx"),
    ]
    palette = plt.get_cmap("tab20")(np.linspace(0.02, 0.92, 12))
    # tab20 has similar light/dark pairs; select well-separated categorical colors.
    palette = plt.get_cmap("tab20")([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5])
    fig, ax = plt.subplots(figsize=(19.5, 10))
    centers = np.arange(1, len(METHOD_ORDER) + 1, dtype=float)
    for method_idx, method in enumerate(METHOD_ORDER):
        color = palette[method_idx]
        for label, df, offset, alpha, hatch in series:
            values = pd.to_numeric(
                df.loc[df["method"].eq(method), "value_mm"], errors="coerce"
            ).dropna().to_numpy(float)
            if not len(values):
                continue
            box = ax.boxplot(
                [values], positions=[centers[method_idx] + offset],
                widths=0.22, patch_artist=True, showfliers=True,
                manage_ticks=False,
                flierprops={"marker": ".", "markersize": 2.5,
                            "markeredgecolor": color, "alpha": 0.3},
            )
            for patch in box["boxes"]:
                patch.set(facecolor=color, edgecolor="#343434", linewidth=0.85,
                          alpha=alpha, hatch=hatch)
            for line in box["whiskers"] + box["caps"]:
                line.set(color="#545454", linewidth=0.9)
            for line in box["medians"]:
                line.set(color="black", linewidth=1.9)

    # Group boundaries, with centered group names below the method labels.
    ranges = group_ranges()
    for idx, group in enumerate(GROUP_ORDER):
        start, end = ranges[group]
        if idx % 2 == 0:
            ax.axvspan(start + 0.5, end + 1.5, facecolor="#f5f6f8", zorder=-1)
        if end < len(METHOD_ORDER) - 1:
            ax.axvline(end + 1.5, color="#c8cbd0", linewidth=0.9, zorder=0)
        center = (start + end + 2) / 2
        ax.text(center, -0.23, group, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10.5, fontweight="bold",
                clip_on=False)

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{m}\n({METHOD_META[m]['year']})" for m in METHOD_ORDER],
                       fontsize=10, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=11, length=0)
    ax.set_xlim(0.45, len(METHOD_ORDER) + 0.55)
    ax.set_ylabel("Distance (mm)", fontsize=12)
    ax.set_title("Baseline, LOSO and Single-reference across 12 methods",
                 fontsize=17, weight="bold", pad=22)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    legend_items = [
        Patch(facecolor="#6f7780", edgecolor="#343434", alpha=0.95,
              label="Baseline"),
        Patch(facecolor="#6f7780", edgecolor="#343434", hatch="//",
              alpha=0.65, label="LOSO"),
        Patch(facecolor="#6f7780", edgecolor="#343434", hatch="xx",
              alpha=0.40, label="Single-reference"),
    ]
    ax.legend(handles=legend_items, ncol=3, loc="upper right", frameon=True,
              fontsize=10, title="Strategy")
    fig.text(0.5, 0.07,
             "Baseline: within-subject localization error. LOSO and Single-reference: "
             "cross-subject raw head-coordinate separation (diagnostic; no alignment).",
             ha="center", va="bottom", fontsize=10, color="#404040")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.31)
    fig.savefig(out / "00_master_three_strategies_boxplot.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_within_boxplot(master: pd.DataFrame, out: Path) -> None:
    baseline, loso = prepare_within_box_data(master)
    fig, ax = plt.subplots(figsize=(16, 8.5))
    centers = np.arange(1, len(METHOD_ORDER) + 1)
    width = 0.32
    add_group_spans_and_labels(ax)

    bp1 = ax.boxplot(
        box_values_by_method(baseline),
        positions=centers - 0.18,
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    style_boxplot(bp1, COLORS["Baseline (historical all-pass)"])

    bp2 = ax.boxplot(
        box_values_by_method(loso),
        positions=centers + 0.18,
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    style_boxplot(bp2, COLORS["LOSO within"])

    ax.set_title("Within-subject anatomical localization error", fontsize=16, fontweight="bold", pad=28)
    ax.set_ylabel("Localization error (mm)")
    ax.set_xlabel("Method (year)")
    ax.set_xticks(centers)
    ax.set_xticklabels([method_label(m) for m in METHOD_ORDER], rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(0.4, len(METHOD_ORDER) + 0.6)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["Baseline (historical all-pass)"], edgecolor=COLORS["Baseline (historical all-pass)"], alpha=0.75, label="Baseline (historical all-pass)"),
            Patch(facecolor=COLORS["LOSO within"], edgecolor=COLORS["LOSO within"], alpha=0.75, label="LOSO within"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.text(0.5, 0.01, "Ordinary vertical boxplots. Values are anatomical localization error in mm.", ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out / "01_within_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_cross_boxplot(master: pd.DataFrame, out: Path) -> None:
    loso_primary, loso_sens, sr_primary, sr_sens = prepare_cross_box_data(master)
    fig, ax = plt.subplots(figsize=(18, 8.8))
    centers = np.arange(1, len(METHOD_ORDER) + 1)
    add_group_spans_and_labels(ax)
    offsets = [-0.27, -0.09, 0.09, 0.27]
    width = 0.15
    series = [
        (loso_primary, COLORS["LOSO cross primary"], "LOSO cross primary", offsets[0]),
        (loso_sens, COLORS["LOSO cross sensitivity"], "LOSO cross sensitivity", offsets[1]),
        (sr_primary, COLORS["Single-reference primary"], "Single-reference primary", offsets[2]),
        (sr_sens, COLORS["Single-reference sensitivity"], "Single-reference sensitivity", offsets[3]),
    ]

    legend_handles = []
    for df, color, label, off in series:
        bp = ax.boxplot(
            box_values_by_method(df),
            positions=centers + off,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        style_boxplot(bp, color)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.75, label=label))

    ax.set_title("Cross-subject no-warp raw head-coordinate separation", fontsize=16, fontweight="bold", pad=28)
    ax.set_ylabel("Raw head-coordinate separation (mm)")
    ax.set_xlabel("Method (year)")
    ax.set_xticks(centers)
    ax.set_xticklabels([method_label(m) for m in METHOD_ORDER], rotation=40, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(0.4, len(METHOD_ORDER) + 0.6)
    ax.legend(handles=legend_handles, loc="upper right", ncol=2, frameon=True)
    fig.text(0.5, 0.01, "Ordinary vertical boxplots. Values are diagnostic raw head-coordinate separation in mm (no warp, no alignment).", ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out / "02_cross_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_pass_rate_grouped(master: pd.DataFrame, out: Path) -> None:
    # derive from run-level rows
    rows = []
    grouped = master.groupby(["scenario", "experiment", "method"], sort=False)
    for (scenario, experiment, method), g in grouped:
        rows.append({
            "scenario": scenario,
            "experiment": experiment,
            "method": method,
            "pass_rate": (g["status"] == "PASS").mean() * 100.0,
        })
    summary = pd.DataFrame(rows)
    summary = summary[summary["method"].isin(METHOD_ORDER)].copy()

    def pick(experiment, scenario):
        return summary[(summary["experiment"] == experiment) & (summary["scenario"] == scenario)].set_index("method").reindex(METHOD_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    centers = np.arange(len(METHOD_ORDER))
    width = 0.36

    left_a = pick("Historical_within", "all_pass_selected")
    left_b = pick("LOSO_within", "primary")
    axes[0].bar(centers - width / 2, left_a["pass_rate"], width=width, color=COLORS["Baseline (historical all-pass)"], label="Baseline")
    axes[0].bar(centers + width / 2, left_b["pass_rate"], width=width, color=COLORS["LOSO within"], label="LOSO within")
    axes[0].set_title("Pass rate — within-subject", fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Pass rate (%)")
    axes[0].set_xticks(centers)
    axes[0].set_xticklabels([method_label(m) for m in METHOD_ORDER], rotation=40, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=True)

    right_a = pick("LOSO_single_reference_no_warp", "primary")
    right_b = pick("LOSO_single_reference_no_warp", "sensitivity")
    right_c = pick("Single-reference_single_reference_no_warp", "primary")
    right_d = pick("Single-reference_single_reference_no_warp", "sensitivity")
    w = 0.18
    axes[1].bar(centers - 1.5 * w, right_a["pass_rate"], width=w, color=COLORS["LOSO cross primary"], label="LOSO primary")
    axes[1].bar(centers - 0.5 * w, right_b["pass_rate"], width=w, color=COLORS["LOSO cross sensitivity"], label="LOSO sensitivity")
    axes[1].bar(centers + 0.5 * w, right_c["pass_rate"], width=w, color=COLORS["Single-reference primary"], label="Single-reference primary")
    axes[1].bar(centers + 1.5 * w, right_d["pass_rate"], width=w, color=COLORS["Single-reference sensitivity"], label="Single-reference sensitivity")
    axes[1].set_title("Pass rate — cross-subject no-warp", fontweight="bold")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(centers)
    axes[1].set_xticklabels([method_label(m) for m in METHOD_ORDER], rotation=40, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=True, ncol=2)

    for ax in axes:
        add_group_spans_and_labels(ax)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out / "03_pass_rate_grouped.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_reference_sensitivity(method_sens: pd.DataFrame, out: Path) -> None:
    df = method_sens[method_sens["scenario"] == "primary"].copy()
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df = df.sort_values("method")
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    axes[0].bar(x, df["median_across_reference_sd_mm"], color="#4E79A7")
    axes[0].set_title("Reference sensitivity: median SD across 6 references", fontweight="bold")
    axes[0].set_ylabel("SD (mm)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([method_label(m) for m in df["method"].astype(str)], rotation=40, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, df["median_across_reference_range_mm"], color="#E15759")
    axes[1].set_title("Reference sensitivity: median range across 6 references", fontweight="bold")
    axes[1].set_ylabel("Range (mm)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([method_label(m) for m in df["method"].astype(str)], rotation=40, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        add_group_spans_and_labels(ax)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out / "04_reference_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def heatmap(ax: plt.Axes, arr: np.ndarray, row_labels: List[str], col_labels: List[str], title: str, cbar_label: str, cmap: str = "viridis"):
    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontweight="bold")
    mean_val = np.nanmean(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                color = "white" if arr[i, j] > mean_val else "black"
                ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=8, color=color)
    return im


def create_reference_by_method_heatmap(ref_method: pd.DataFrame, out: Path) -> None:
    df = ref_method[ref_method["scenario"] == "primary"].copy()
    ref_order = [f"sub-0{i}" for i in range(1, 8)]
    arr = np.full((len(ref_order), len(METHOD_ORDER)), np.nan)
    for i, ref in enumerate(ref_order):
        for j, method in enumerate(METHOD_ORDER):
            sub = df[(df["reference_subject"] == ref) & (df["method"] == method)]
            if not sub.empty:
                arr[i, j] = float(sub["median_mm"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 6))
    im = heatmap(ax, arr, ref_order, METHOD_ORDER, "Median raw separation by reference and method", "Median raw separation (mm)", cmap="magma")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median raw separation (mm)")
    fig.tight_layout()
    fig.savefig(out / "05_reference_by_method_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_test_reference_heatmap(pairing: pd.DataFrame, out: Path) -> None:
    df = pairing[pairing["scenario"] == "primary"].copy()
    ref_order = [f"sub-0{i}" for i in range(1, 8)]
    test_order = [f"sub-0{i}" for i in range(1, 8)]
    agg = df.groupby(["test_subject", "reference_subject"], as_index=False)["median_mm"].median()
    arr = np.full((len(test_order), len(ref_order)), np.nan)
    for i, test in enumerate(test_order):
        for j, ref in enumerate(ref_order):
            sub = agg[(agg["test_subject"] == test) & (agg["reference_subject"] == ref)]
            if not sub.empty:
                arr[i, j] = float(sub["median_mm"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = heatmap(ax, arr, test_order, ref_order, "Median raw separation by test-reference pair", "Median raw separation (mm)", cmap="plasma")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median raw separation (mm)")
    fig.tight_layout()
    fig.savefig(out / "06_test_reference_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_manifest(out: Path, args: argparse.Namespace, master: pd.DataFrame, ref_method: pd.DataFrame, method_sens: pd.DataFrame, pairing: pd.DataFrame) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "master_run": str(args.master_run),
            "reference_method": str(args.reference_method),
            "method_sensitivity": str(args.method_sensitivity),
            "pairing": str(args.pairing),
        },
        "row_counts": {
            "master_run": int(len(master)),
            "reference_method": int(len(ref_method)),
            "method_sensitivity": int(len(method_sens)),
            "pairing": int(len(pairing)),
        },
        "note": "No warp / no alignment preserved. Within-subject figures use anatomical localization error; cross-subject figures use diagnostic raw head-coordinate separation.",
        "outputs": [
            "00_master_three_strategies_boxplot.png",
            "01_within_boxplot.png",
            "02_cross_boxplot.png",
            "03_pass_rate_grouped.png",
            "04_reference_sensitivity.png",
            "05_reference_by_method_heatmap.png",
            "06_test_reference_heatmap.png",
        ],
    }
    (out / "07_plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        "Script 09 outputs (PNG only)\n"
        "============================\n\n"
        "00_master_three_strategies_boxplot.png: Baseline, LOSO and Single-reference grouped by method, each method colored differently.\n"
        "01_within_boxplot.png: Baseline vs LOSO within-subject vertical boxplots.\n"
        "02_cross_boxplot.png: LOSO and single-reference cross-subject no-warp vertical boxplots.\n"
        "03_pass_rate_grouped.png: Pass-rate comparison across strategies.\n"
        "04_reference_sensitivity.png: Median SD and range across six references.\n"
        "05_reference_by_method_heatmap.png: Median raw separation by reference and method.\n"
        "06_test_reference_heatmap.png: Median raw separation by test-reference pair.\n\n"
        "Interpretation note:\n"
        "- Baseline and LOSO within use anatomical localization error (mm).\n"
        "- Cross-subject no-warp uses diagnostic raw head-coordinate separation (mm).\n",
        encoding="utf-8",
    )


def audit(master: pd.DataFrame, ref_method: pd.DataFrame, method_sens: pd.DataFrame, pairing: pd.DataFrame) -> None:
    expected_master = 12444  # 732 historical + 5856 primary + 5856 sensitivity
    if len(master) != expected_master:
        raise ValueError(f"Expected {expected_master} master run-level rows, got {len(master)}")
    if len(ref_method[ref_method["scenario"] == "primary"]) != 84:
        raise ValueError("Expected 84 primary rows in 02_reference_method_summary.csv")
    if len(method_sens[method_sens["scenario"] == "primary"]) != 12:
        raise ValueError("Expected 12 primary rows in 04_method_sensitivity_summary.csv")
    if len(pairing[pairing["scenario"] == "primary"]) != 504:
        raise ValueError("Expected 504 primary rows in 05_reference_pairing_matrix.csv")
    print(f"AUDIT PASSED | master_run={len(master)} | ref_method={len(ref_method)} | method_sensitivity={len(method_sens)} | pairing={len(pairing)}")


def main() -> None:
    args = parse_args()
    validate_inputs([args.master_run, args.reference_method, args.method_sensitivity, args.pairing])
    master, ref_method, method_sens, pairing = load_data(args)
    audit(master, ref_method, method_sens, pairing)
    if args.phase == "audit":
        return
    ensure_output_dir(args.output_root, args.overwrite)
    create_master_three_strategy_boxplot(master, args.output_root)
    create_within_boxplot(master, args.output_root)
    create_cross_boxplot(master, args.output_root)
    create_pass_rate_grouped(master, args.output_root)
    create_reference_sensitivity(method_sens, args.output_root)
    create_reference_by_method_heatmap(ref_method, args.output_root)
    create_test_reference_heatmap(pairing, args.output_root)
    write_manifest(args.output_root, args, master, ref_method, method_sens, pairing)
    print(f"Plots written: {args.output_root}")


if __name__ == "__main__":
    main()
