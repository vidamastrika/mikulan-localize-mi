#!/usr/bin/env python3
"""Create PNG figures from completed strategy-comparison and sensitivity tables.

Sample terminal inputs (from the repository root):
    python cross_subject_warp/scripts/09_visualize_results.py --phase audit
    python cross_subject_warp/scripts/09_visualize_results.py \
      --comparison-root cross_subject_warp/outputs/07_compare_strategies_review
    python cross_subject_warp/scripts/09_visualize_results.py \
      --comparison-root cross_subject_warp/outputs/07_compare_strategies_review \
      --output-root cross_subject_warp/outputs/09_visualize_results_review

This is a plotting-only stage. It does not fit registrations, reconstruct
sources, change coordinates, or recalculate localization errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METHODS = ["MNE", "dSPM", "sLORETA", "eLORETA", "Continuous-ECD", "RAP-MUSIC",
           "LCMV", "MxNE", "irMxNE", "Gamma-MAP", "Champagne", "SESAME"]
METHOD_META = {
    "MNE": ("Minimum-norm", "1994"), "dSPM": ("Minimum-norm", "2002"),
    "sLORETA": ("Minimum-norm", "2002"), "eLORETA": ("Minimum-norm", "2011"),
    "Continuous-ECD": ("Dipole / Beamformer", "1990"), "RAP-MUSIC": ("Dipole / Beamformer", "2001"),
    "LCMV": ("Dipole / Beamformer", "1997"), "MxNE": ("Sparse", "2011"),
    "irMxNE": ("Sparse", "2018"), "Gamma-MAP": ("Bayesian", "2008"),
    "Champagne": ("Bayesian", "2016"), "SESAME": ("SESAME", "2020"),
}
METHOD_GROUPS = ["Minimum-norm", "Dipole / Beamformer", "Sparse", "Bayesian", "SESAME"]
POLICIES = ["primary_rt010", "pair_rt015_source_rt010"]
MODES = ["affine", "syn"]
COLORS = {"historical_baseline": "#4E79A7", "no_warp": "#F28E2B", "warp_primary": "#59A14F",
          "warp_sensitivity": "#8CD17D", "LOSO": "#59A14F", "Single-reference": "#E15759"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_metric(df: pd.DataFrame) -> pd.Series:
    return df.localization_error_valid.astype(str).str.lower().isin(["true", "1"]) & pd.to_numeric(df.localization_error_mm, errors="coerce").notna()


def validate_inputs(comparison_root: Path, reference_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs_path = comparison_root / "01_localization_observations.csv"
    matched_path = comparison_root / "05_loso_vs_single_reference_matched.csv"
    ref_path = reference_root / "02_reference_method_summary.csv"
    run_path = reference_root / "03_test_run_sensitivity.csv"
    paths = [obs_path, matched_path, ref_path, run_path]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    obs, matched, ref, perrun = [pd.read_csv(path, low_memory=False) for path in paths]
    required_obs = {"comparison_source", "strategy", "peak_scenario", "qa_policy", "condition", "registration_mode",
                    "method", "test_subject", "run", "localization_error_mm", "localization_error_valid"}
    if not required_obs.issubset(obs.columns):
        raise ValueError(f"Comparison observations missing {sorted(required_obs-set(obs.columns))}")
    required_ref = {"qa_policy", "peak_scenario", "registration_mode", "method", "reference_subject", "median_mm", "scheduled", "mapped_valid"}
    if not required_ref.issubset(ref.columns):
        raise ValueError(f"Reference summary missing {sorted(required_ref-set(ref.columns))}")
    required_run = {"qa_policy", "peak_scenario", "registration_mode", "method", "test_subject", "run", "std_mm", "mapped_valid"}
    if not required_run.issubset(perrun.columns):
        raise ValueError(f"Test-run sensitivity summary missing {sorted(required_run-set(perrun.columns))}")
    required_match = {"registration_mode", "qa_policy", "peak_scenario", "method", "matched_valid", "single_reference_minus_loso_mm", "_merge"}
    if not required_match.issubset(matched.columns):
        raise ValueError(f"Matched comparison missing {sorted(required_match-set(matched.columns))}")
    return obs, matched, ref, perrun


def boxplot_by_method(ax, df: pd.DataFrame, value: str, series: list[tuple[str, str, pd.DataFrame]], title: str,
                      ylabel: str = "Localization error (mm)") -> None:
    positions, values, colors, labels = [], [], [], []
    width = .8 / max(1, len(series))
    for si, (name, color, frame) in enumerate(series):
        for mi, method in enumerate(METHODS):
            x = frame.loc[frame.method.eq(method), value]
            x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(x):
                positions.append(mi + 1 + (si - (len(series)-1)/2) * width)
                values.append(x)
                colors.append(color)
                labels.append(name)
    if values:
        artists = ax.boxplot(values, positions=positions, widths=width*.82, patch_artist=True,
                             showfliers=False, manage_ticks=False)
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(.72)
        for part in ("whiskers", "caps", "medians"):
            for artist in artists[part]: artist.set_color("#333333")
    ax.set_xticks(range(1, len(METHODS)+1), METHODS, rotation=55, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=.22)


def plot_within(obs: pd.DataFrame, out: Path) -> None:
    hist = obs[(obs.comparison_source == "historical_baseline") & (obs.peak_scenario == "primary")]
    nw = obs[(obs.comparison_source == "no_warp") & (obs.condition == "within") & (obs.peak_scenario == "primary")]
    warped = obs[(obs.comparison_source == "warp_primary") & (obs.condition == "within") &
                 (obs.peak_scenario == "primary") & (obs.qa_policy == "primary_rt010")]
    for frame in (hist, nw, warped):
        frame["localization_error_valid"] = valid_metric(frame)
    fig, ax = plt.subplots(figsize=(15, 7))
    boxplot_by_method(ax, hist[hist.localization_error_valid], "localization_error_mm", [
        ("Historical", COLORS["historical_baseline"], hist[hist.localization_error_valid]),
        ("No-warp within", COLORS["no_warp"], nw[nw.localization_error_valid]),
        ("Warp within", COLORS["warp_primary"], warped[warped.localization_error_valid]),
    ], "Within-subject localization error; historical baseline and completed evaluations")
    ax.legend(handles=[Patch(facecolor=COLORS[k], label=label)
                       for k, label in [("historical_baseline", "Historical"), ("no_warp", "No-warp within"), ("warp_primary", "Warp within")]],
              loc="upper right", ncol=3)
    fig.tight_layout(); fig.savefig(out / "01_within_localization_error.png", dpi=180); plt.close(fig)


def plot_cross_errors(obs: pd.DataFrame, out: Path) -> None:
    d = obs[obs.comparison_source.isin(["warp_primary", "warp_sensitivity"]) & obs.condition.eq("cross_reference") &
            obs.peak_scenario.eq("primary")].copy()
    d["valid"] = valid_metric(d)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
    for i, policy in enumerate(POLICIES):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            panel = d[d.qa_policy.eq(policy) & d.registration_mode.eq(mode) & d.valid]
            boxplot_by_method(ax, panel, "localization_error_mm", [
                ("LOSO", COLORS["LOSO"], panel[panel.strategy.eq("LOSO")]),
                ("Single-reference", COLORS["Single-reference"], panel[panel.strategy.eq("Single-reference")]),
            ], f"{policy}; {mode}; primary peak")
    handles = [Patch(facecolor=COLORS[k], label=k) for k in ["LOSO", "Single-reference"]]
    fig.legend(handles=handles, loc="upper center", ncol=2); fig.suptitle("Mapped cross-subject localization errors (valid observations only)", y=1.01)
    fig.tight_layout(); fig.savefig(out / "02_cross_subject_localization_error.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_coverage(ref: pd.DataFrame, out: Path) -> None:
    d = ref[ref.peak_scenario.eq("primary")].groupby(["qa_policy", "registration_mode", "method"], as_index=False).agg(
        scheduled=("scheduled", "sum"), valid=("mapped_valid", "sum"))
    d["coverage"] = d.valid / d.scheduled
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    x = np.arange(len(METHODS)); width = .38
    for ax, policy in zip(axes, POLICIES):
        q = d[d.qa_policy.eq(policy)]
        for k, mode in enumerate(MODES):
            g = q[q.registration_mode.eq(mode)].set_index("method").reindex(METHODS)
            ax.bar(x + (k-.5)*width, g.coverage, width, label=mode)
        ax.set_title(policy); ax.set_xticks(x, METHODS, rotation=55, ha="right"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Valid mapped observations / scheduled"); axes[-1].legend(title="Registration")
    fig.suptitle("Primary-peak mapping coverage by method and QA policy"); fig.tight_layout()
    fig.savefig(out / "03_mapping_coverage.png", dpi=180); plt.close(fig)


def plot_reference_spread(perrun: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
    for i, policy in enumerate(POLICIES):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            d = perrun[(perrun.qa_policy == policy) & (perrun.registration_mode == mode) &
                       (perrun.peak_scenario == "primary")]
            vals, pos = [], []
            for k, method in enumerate(METHODS, 1):
                x = pd.to_numeric(d.loc[d.method.eq(method), "std_mm"], errors="coerce").dropna().to_numpy()
                if len(x): vals.append(x); pos.append(k)
            if vals:
                ax.boxplot(vals, positions=pos, widths=.68, showfliers=False)
            ax.set_xticks(range(1, len(METHODS)+1), METHODS, rotation=55, ha="right")
            ax.set_title(f"{policy}; {mode}"); ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Across-reference SD per test-run (mm)"); axes[1, 0].set_ylabel("Across-reference SD per test-run (mm)")
    fig.suptitle("Reference-choice sensitivity across six references (primary peak)"); fig.tight_layout()
    fig.savefig(out / "04_reference_sensitivity.png", dpi=180); plt.close(fig)


def plot_reference_heatmaps(ref: pd.DataFrame, out: Path) -> None:
    d = ref[ref.peak_scenario.eq("primary")]
    references = sorted(d.reference_subject.dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(19, 12), sharex=True, sharey=True)
    image = None
    for i, policy in enumerate(POLICIES):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            q = d[d.qa_policy.eq(policy) & d.registration_mode.eq(mode)]
            table = q.pivot(index="method", columns="reference_subject", values="median_mm").reindex(index=METHODS, columns=references)
            image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=100)
            ax.set_title(f"{policy}; {mode}"); ax.set_xticks(range(len(references)), references, rotation=35, ha="right")
            ax.set_yticks(range(len(METHODS)), METHODS)
            ax.set_xticks(np.arange(-.5, len(references), 1), minor=True); ax.set_yticks(np.arange(-.5, len(METHODS), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.4); ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Median localization error (mm)", shrink=.82)
    fig.suptitle("Single-reference median localization error by reference (primary peak)")
    fig.savefig(out / "05_reference_by_method_heatmap.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_loso_sr(matched: pd.DataFrame, out: Path) -> None:
    d = matched[(matched._merge == "both") & matched.peak_scenario.eq("primary") &
                matched.matched_valid.astype(str).str.lower().isin(["true", "1"])].copy()
    d["delta"] = pd.to_numeric(d.single_reference_minus_loso_mm, errors="coerce")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
    for i, policy in enumerate(POLICIES):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]; q = d[d.qa_policy.eq(policy) & d.registration_mode.eq(mode)]
            vals, pos = [], []
            for k, method in enumerate(METHODS, 1):
                x = q.loc[q.method.eq(method), "delta"].dropna().to_numpy()
                if len(x): vals.append(x); pos.append(k)
            if vals: ax.boxplot(vals, positions=pos, widths=.68, showfliers=False)
            ax.axhline(0, color="#555", linewidth=.8); ax.set_xticks(range(1, len(METHODS)+1), METHODS, rotation=55, ha="right")
            ax.set_title(f"{policy}; {mode}"); ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Single-reference minus LOSO (mm)"); axes[1, 0].set_ylabel("Single-reference minus LOSO (mm)")
    fig.suptitle("Matched mapped localization errors at identical reference (primary peak)")
    fig.tight_layout(); fig.savefig(out / "06_loso_vs_single_reference_matched.png", dpi=180); plt.close(fig)


def master_affine_syn_rows(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact common baseline/affine/SyN LOSO cases for each QA policy."""
    keys = ["method", "test_subject", "run"]
    baseline = obs[(obs.comparison_source == "historical_baseline") & obs.peak_scenario.eq("primary")].copy()
    baseline = baseline[keys + ["localization_error_mm", "localization_error_valid"]].rename(columns={
        "localization_error_mm": "baseline_error_mm", "localization_error_valid": "baseline_valid"})
    if baseline.duplicated(keys).any():
        raise ValueError("Historical baseline is not unique by method/test/run")
    datasets, coverage = [], []
    for policy, source in [("primary_rt010", "warp_primary"), ("pair_rt015_source_rt010", "warp_sensitivity")]:
        d = obs[(obs.comparison_source == source) & obs.strategy.eq("LOSO") & obs.condition.eq("cross_reference") &
                obs.scenario.eq("primary") & obs.peak_scenario.eq("primary") & obs.qa_policy.eq(policy)].copy()
        if d.empty:
            raise ValueError(f"No LOSO primary-peak records for {policy}")
        mode_tables = {}
        for mode in MODES:
            m = d[d.registration_mode.eq(mode)][keys + ["reference_subject", "localization_error_mm", "localization_error_valid"]].copy()
            if m.duplicated(keys).any():
                raise ValueError(f"{policy}/{mode}: duplicate LOSO method/test/run observations")
            m = m.rename(columns={"reference_subject": f"{mode}_reference_subject",
                                  "localization_error_mm": f"{mode}_error_mm",
                                  "localization_error_valid": f"{mode}_valid"})
            mode_tables[mode] = m
        pair = mode_tables["affine"].merge(mode_tables["syn"], on=keys, how="outer", validate="one_to_one", indicator=True)
        if not pair._merge.eq("both").all():
            raise ValueError(f"{policy}: affine/SyN scheduled observation keys differ")
        if not pair.affine_reference_subject.eq(pair.syn_reference_subject).all():
            raise ValueError(f"{policy}: affine/SyN reference subjects differ")
        pair = pair.merge(baseline, on=keys, how="left", validate="one_to_one", indicator="baseline_merge")
        if not pair.baseline_merge.eq("both").all():
            raise ValueError(f"{policy}: historical baseline missing for a LOSO observation")
        for c in ["baseline_valid", "affine_valid", "syn_valid"]:
            pair[c] = pair[c].astype(str).str.lower().isin(["true", "1"])
        common = pair.baseline_valid & pair.affine_valid & pair.syn_valid
        pair["qa_policy"] = policy
        pair["common_valid"] = common
        counts = pair.groupby("method", sort=False).agg(
            scheduled=("run", "size"), baseline_valid=("baseline_valid", "sum"),
            affine_valid=("affine_valid", "sum"), syn_valid=("syn_valid", "sum"),
            common_valid=("common_valid", "sum")).reset_index()
        counts["qa_policy"] = policy
        coverage.append(counts)
        datasets.append(pair.loc[common, keys + ["affine_reference_subject", "qa_policy", "baseline_error_mm", "affine_error_mm", "syn_error_mm"]])
    return pd.concat(datasets, ignore_index=True), pd.concat(coverage, ignore_index=True)


def _group_positions(ax: plt.Axes) -> None:
    ranges = []
    start = 0
    for group in METHOD_GROUPS:
        n = sum(1 for m in METHODS if METHOD_META[m][0] == group)
        ranges.append((group, start, start+n-1))
        start += n
    for gi, (group, start, end) in enumerate(ranges):
        if gi % 2 == 0:
            ax.axvspan(start + .5, end + 1.5, color="#f5f6f8", zorder=-1)
        if gi < len(ranges)-1:
            ax.axvline(end + 1.5, color="#c8cbd0", linewidth=.8, zorder=0)
        ax.text((start + end + 2)/2, -.20, group, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, fontweight="bold", clip_on=False)


def plot_master_baseline_affine_syn(data: pd.DataFrame, out: Path) -> None:
    """Master figure: matched historical baseline and LOSO affine/SyN errors."""
    series = [("baseline_error_mm", "Baseline", .95, ""),
              ("affine_error_mm", "Affine", .68, "//"),
              ("syn_error_mm", "SyN", .48, "xx")]
    palette = plt.get_cmap("tab20")([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5])
    fig, axes = plt.subplots(2, 1, figsize=(20, 15), sharex=True, sharey=True)
    for ax, policy in zip(axes, POLICIES):
        q = data[data.qa_policy.eq(policy)]
        _group_positions(ax)
        for mi, method in enumerate(METHODS):
            g = q[q.method.eq(method)]
            for si, (column, label, alpha, hatch) in enumerate(series):
                vals = pd.to_numeric(g[column], errors="coerce").dropna().to_numpy(float)
                if not len(vals):
                    continue
                bp = ax.boxplot([vals], positions=[mi+1+(si-1)*.24], widths=.21,
                                patch_artist=True, showfliers=True, manage_ticks=False,
                                flierprops={"marker": ".", "markersize": 2.2,
                                            "markeredgecolor": palette[mi], "alpha": .3})
                for box in bp["boxes"]:
                    box.set(facecolor=palette[mi], edgecolor="#343434", linewidth=.8, alpha=alpha, hatch=hatch)
                for line in bp["whiskers"] + bp["caps"]:
                    line.set(color="#555555", linewidth=.8)
                for line in bp["medians"]:
                    line.set(color="black", linewidth=1.7)
        ax.set_ylabel("Localization error (mm)")
        title = "PRIMARY QA (pair and source-point roundtrip ≤ 0.1 mm)" if policy == POLICIES[0] else "QA SENSITIVITY (pair ≤ 0.15 mm; source-point ≤ 0.1 mm)"
        n_min = int(q.groupby("method").size().min()) if len(q) else 0
        n_max = int(q.groupby("method").size().max()) if len(q) else 0
        ax.set_title(f"{title} | matched valid cases per method: {n_min}–{n_max}", fontsize=11, loc="left")
        ax.set_xlim(.45, len(METHODS)+.55); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[-1].set_xticks(np.arange(1, len(METHODS)+1))
    axes[-1].set_xticklabels([f"{m}\n({METHOD_META[m][1]})" for m in METHODS], fontsize=9)
    axes[-1].tick_params(axis="x", pad=8, length=0)
    handles = [Patch(facecolor="#737373", edgecolor="#343434", alpha=alpha, hatch=hatch, label=label)
               for _, label, alpha, hatch in series]
    fig.legend(handles=handles, loc="upper right", ncol=3, title="Matched observation condition")
    fig.suptitle("Historical baseline versus LOSO affine and SyN mapping", fontsize=17, weight="bold", y=.995)
    fig.text(.5, .018,
             "Baseline is historical within-subject error; Affine and SyN are LOSO cross-subject errors in test anatomy. "
             "Each box uses only method/test/run cases valid under all three conditions within that QA policy. "
             "The lower panel is a separate QA sensitivity, not the primary result.",
             ha="center", va="bottom", fontsize=9, color="#444444", wrap=True)
    fig.subplots_adjust(left=.065, right=.99, top=.955, bottom=.13, hspace=.30)
    fig.savefig(out / "00_master_baseline_affine_syn.png", dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _distribution(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not len(x):
        return dict(N=0, Mean_mm=np.nan, Median_mm=np.nan, SD_mm=np.nan, Q1_mm=np.nan,
                    Q3_mm=np.nan, IQR_mm=np.nan, Min_mm=np.nan, Max_mm=np.nan)
    q1, q3 = x.quantile([.25, .75]).tolist()
    return dict(N=int(len(x)), Mean_mm=float(x.mean()), Median_mm=float(x.median()),
                SD_mm=float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                Q1_mm=float(q1), Q3_mm=float(q3), IQR_mm=float(q3-q1),
                Min_mm=float(x.min()), Max_mm=float(x.max()))


def master_figure_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Statistics on exactly the common valid observations shown in the master figure."""
    rows = []
    series = [("Baseline", "none", "baseline_error_mm"),
              ("LOSO", "affine", "affine_error_mm"),
              ("LOSO", "syn", "syn_error_mm")]
    for policy in POLICIES:
        for method in METHODS:
            g = data[(data.qa_policy == policy) & data.method.eq(method)]
            for strategy, mode, column in series:
                row = {"method": method, "strategy": strategy, "registration_mode": mode,
                       "qa_policy": policy, "figure_series": "Baseline" if mode == "none" else mode.capitalize(),
                       "cohort": "common valid baseline/affine/SyN method-test-run cases"}
                row.update(_distribution(g[column]))
                rows.append(row)
    return pd.DataFrame(rows)


def method_strategy_statistics(obs: pd.DataFrame) -> pd.DataFrame:
    """Comparable to the no-warp master statistics, with mode/policy axes added."""
    hist = obs[(obs.comparison_source == "historical_baseline") & obs.peak_scenario.eq("primary")].copy()
    hist["strategy_summary"] = "Baseline"
    hist["registration_mode_summary"] = "none"
    hist["qa_policy_summary"] = "not_applicable"
    hist["peak_summary"] = "primary"
    hist["metric_basis"] = "historical within-subject localization error"
    mapped = obs[obs.comparison_source.isin(["warp_primary", "warp_sensitivity"]) &
                 obs.condition.eq("cross_reference")].copy()
    mapped["strategy_summary"] = mapped.strategy
    mapped["registration_mode_summary"] = mapped.registration_mode
    mapped["qa_policy_summary"] = mapped.qa_policy
    mapped["peak_summary"] = mapped.peak_scenario
    mapped["metric_basis"] = "mapped cross-subject localization error in test anatomy"
    df = pd.concat([hist, mapped], ignore_index=True, sort=False)
    df["valid_for_stats"] = valid_metric(df)
    keys = ["method", "strategy_summary", "registration_mode_summary", "qa_policy_summary", "peak_summary", "metric_basis"]
    rows = []
    for key, g in df.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        valid = g.valid_for_stats
        row.update(Scheduled=int(len(g)), Valid_N=int(valid.sum()), Invalid_N=int((~valid).sum()),
                   Coverage=float(valid.mean()) if len(g) else np.nan,
                   **_distribution(g.loc[valid, "localization_error_mm"]))
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"strategy_summary": "strategy",
        "registration_mode_summary": "registration_mode", "qa_policy_summary": "qa_policy",
        "peak_summary": "peak_scenario"})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-root", default="cross_subject_warp/outputs/07_compare_strategies_review")
    p.add_argument("--reference-root", default="cross_subject_warp/outputs/08_reference_sensitivity")
    p.add_argument("--output-root", default="cross_subject_warp/outputs/09_visualize_results")
    p.add_argument("--phase", choices=["audit", "plot"], default="plot")
    a = p.parse_args(argv)
    comparison = (ROOT / a.comparison_root).resolve()
    reference = (ROOT / a.reference_root).resolve()
    out = (ROOT / a.output_root).resolve()
    if a.phase == "plot" and out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    obs, matched, ref, perrun = validate_inputs(comparison, reference)
    master_data, master_coverage = master_affine_syn_rows(obs)
    matched_rows = matched[(matched._merge == "both") & matched.peak_scenario.eq("primary")]
    if matched_rows.empty:
        raise ValueError("No matched LOSO/single-reference rows. Regenerate Script 07 outputs with the corrected matching key first.")
    if a.phase == "audit":
        print(f"AUDIT PASSED: observations={len(obs)}; reference summaries={len(ref)}; matched primary rows={len(matched_rows)}")
        print("Common valid baseline/affine/SyN cases by policy and method:")
        print(master_coverage[["qa_policy", "method", "scheduled", "common_valid"]].to_string(index=False))
        print("No plots were written.")
        return 0
    out.mkdir(parents=True, exist_ok=False)
    plot_within(obs, out)
    plot_cross_errors(obs, out)
    plot_coverage(ref, out)
    plot_reference_spread(perrun, out)
    plot_reference_heatmaps(ref, out)
    plot_loso_sr(matched, out)
    plot_master_baseline_affine_syn(master_data, out)
    master_data.to_csv(out / "00_master_baseline_affine_syn_plot_data.csv", index=False)
    master_coverage.to_csv(out / "00_master_baseline_affine_syn_coverage.csv", index=False)
    master_stats = master_figure_statistics(master_data)
    strategy_stats = method_strategy_statistics(obs)
    master_stats.to_csv(out / "00_master_baseline_affine_syn_statistics.csv", index=False)
    strategy_stats.to_csv(out / "08_method_strategy_mode_statistics.csv", index=False)
    files = sorted(p.name for p in out.glob("*.png"))
    input_paths = [comparison / "01_localization_observations.csv", comparison / "05_loso_vs_single_reference_matched.csv",
                   reference / "02_reference_method_summary.csv", reference / "03_test_run_sensitivity.csv"]
    manifest = {
        "script": Path(__file__).name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": {str(p): sha256(p) for p in input_paths},
        "input_rows": {"localization_observations": len(obs), "strategy_matches": len(matched),
                       "reference_method_summary": len(ref), "test_run_sensitivity": len(perrun)},
        "plots": files,
        "master_figure_common_valid_rows": len(master_data),
        "master_figure_coverage_rows": len(master_coverage),
        "master_figure_statistics_rows": len(master_stats),
        "method_strategy_statistics_rows": len(strategy_stats),
        "interpretation": ["Only valid mapped localization errors are plotted.",
                           "QA policies and registration modes are kept separate.",
                           "Reference sensitivity changes both reference anatomy and reference-selected parameters.",
                           "Plots describe numeric localization outcomes and do not establish anatomical accuracy."],
        "registration_fitting_executed": False,
        "inverse_reconstruction_executed": False,
    }
    (out / "07_plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "README.txt").write_text("Script 09 visualization outputs (PNG only)\n\n" + "\n".join(files) +
        "\n\nQA policies and registration modes remain separate. Invalid observations are excluded from error distributions but are represented in the coverage figure. These figures do not establish anatomical accuracy.\n")
    print(f"Plots written: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
