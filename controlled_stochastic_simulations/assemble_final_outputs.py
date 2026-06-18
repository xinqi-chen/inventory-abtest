#!/usr/bin/env python3
"""Assemble the final controlled-stochastic simulation artifacts.

This script takes the separately reproduced Scenario 1 and Scenario 2 output
directories and writes a compact final folder with selected-regime summaries,
raw selected estimates, warnings, and a Markdown report.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


SELECTED_CAPACITIES = {
    1: [("tight", 0.90), ("medium", 0.92), ("loose", 1.20)],
    2: [("tight", 0.85), ("medium", 1.00), ("loose", 1.10)],
}

DESIGN_ORDER = ("SW", "IR", "BR")
DISPLAY_DESIGN = {"SW": "SW", "IR": "IR", "BR": "PR"}


def close_to(series: pd.Series, value: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").sub(value).abs() < 1e-9


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def selected_reason(scenario: int, label: str, cf: float) -> str:
    if scenario == 1 and label == "medium" and abs(cf - 0.92) < 1e-9:
        return "manual replacement: cf=0.92 keeps IR/PR capacity binding while less tight than cf=0.91"
    if scenario == 1:
        return "selected controlled-stochastic regime for final theory verification"
    return "selected high-signal Scenario 2 regime for final theory verification"


def build_selected(summary: pd.DataFrame, scenario: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label, cf in SELECTED_CAPACITIES[scenario]:
        sub = summary[close_to(summary["capacity_factor"], cf)]
        if sub.empty:
            raise ValueError(f"Scenario {scenario}: missing selected capacity factor {cf:g}")

        by_design = {str(row["design"]): row for _, row in sub.iterrows()}
        missing = [design for design in DESIGN_ORDER if design not in by_design]
        if missing:
            raise ValueError(
                f"Scenario {scenario}, cf={cf:g}: missing designs {', '.join(missing)}"
            )

        first = sub.iloc[0]
        row: Dict[str, object] = {
            "scenario": scenario,
            "label": label,
            "N": float(first["N"]),
            "capacity_factor": cf,
            "reason_for_selection": selected_reason(scenario, label, cf),
            "GTE": float(first["GTE"]),
            "bias_SW": float(by_design["SW"]["bias"]),
            "bias_IR": float(by_design["IR"]["bias"]),
            "bias_BR": float(by_design["BR"]["bias"]),
            "leftover_delta": first.get("leftover_delta", ""),
            "bias_BR_minus_SW": "",
            "capacity_binding_rate_SW": float(by_design["SW"]["capacity_binding_rate"]),
            "capacity_binding_rate_IR": float(by_design["IR"]["capacity_binding_rate"]),
            "capacity_binding_rate_BR": float(by_design["BR"]["capacity_binding_rate"]),
            "margin_violation_rate_max": float(sub["margin_violation_rate"].max()),
            "lower_bound_active_rate_max": float(sub["lower_bound_active_rate"].max()),
            "no_overshoot_violation_rate_max": float(sub["no_overshoot_violation_rate"].max()),
        }
        if scenario == 2:
            row["leftover_delta"] = ""
            row["bias_BR_minus_SW"] = row["bias_BR"] - row["bias_SW"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_selected_raw(run: pd.DataFrame, selected: pd.DataFrame, scenario: int, path: Path) -> None:
    pieces: List[pd.DataFrame] = []
    for _, pick in selected[selected["scenario"] == scenario].iterrows():
        label = str(pick["label"])
        cf = float(pick["capacity_factor"])
        n = float(pick["N"])
        sub = run[
            close_to(run["capacity_factor"], cf)
            & (pd.to_numeric(run["N"], errors="coerce").sub(n).abs() < 1e-9)
            & run["design"].isin(DESIGN_ORDER)
        ].copy()
        sub["selected_label"] = label
        pieces.append(sub)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out.to_csv(path, index=False)


def copy_source_artifacts(src: Path, dst: Path, scenario: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    names = [
        f"run_level_results_scenario{scenario}.csv",
        f"summary_scenario{scenario}_full_grid.csv",
        f"raw_estimates_scenario{scenario}_selected.csv",
        "selected_capacity_regimes.csv",
        "theory_region_warnings.txt",
        "simulation_metadata.json",
    ]
    for name in names:
        copy_if_exists(src / name, dst / name)


def copy_figures(src_dirs: Iterable[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src in src_dirs:
        fig_dir = src / "figs_abtest_style"
        if fig_dir.exists():
            for path in fig_dir.glob("*.png"):
                copy_if_exists(path, dst / path.name)


def format_float(value: object, digits: int = 3) -> str:
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(selected: pd.DataFrame, scenario: int) -> str:
    cols = [
        "Regime",
        "cf",
        "GTE",
        "SW bias",
        "IR bias",
        "PR bias",
        "SW bind",
        "IR bind",
        "PR bind",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in selected[selected["scenario"] == scenario].iterrows():
        vals = [
            str(row["label"]),
            format_float(row["capacity_factor"], 2),
            format_float(row["GTE"]),
            format_float(row["bias_SW"]),
            format_float(row["bias_IR"]),
            format_float(row["bias_BR"]),
            format_float(row["capacity_binding_rate_SW"]),
            format_float(row["capacity_binding_rate_IR"]),
            format_float(row["capacity_binding_rate_BR"]),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def diagnostics_pass(selected: pd.DataFrame) -> bool:
    diag_cols = [
        "margin_violation_rate_max",
        "lower_bound_active_rate_max",
        "no_overshoot_violation_rate_max",
    ]
    return bool((selected[diag_cols].fillna(0.0) == 0.0).all().all())


def write_warnings(selected: pd.DataFrame, path: Path) -> None:
    diag_cols = [
        "margin_violation_rate_max",
        "lower_bound_active_rate_max",
        "no_overshoot_violation_rate_max",
    ]
    failures = selected[(selected[diag_cols].fillna(0.0) != 0.0).any(axis=1)]
    if failures.empty:
        text = "No selected-regime numerical validity check failures were detected.\n"
    else:
        text = "WARNING: selected-regime numerical validity check failures detected.\n"
        text += failures.to_csv(index=False)
    path.write_text(text, encoding="utf-8")


def write_report(selected: pd.DataFrame, output_dir: Path) -> None:
    status = "pass" if diagnostics_pass(selected) else "fail"
    text = f"""# Controlled Stochastic Simulation Reproduction Report

This folder contains the final controlled stochastic simulations used to verify the uniform-demand theory separately from the trace-driven FreshRetailNet experiments.

## Code and Estimator

- Randomization designs: switchback (SW), item-level randomization (IR), and paired/item-matched randomization. The simulator keeps the historical internal code name `BR`, while all figures and report text display this design as `PR`.
- Treatment probability: `p = 0.5`.
- Assignment mode: strict Bernoulli assignment for the final runs.
- Reported estimator: difference in means, using the simulator column `diff_in_means`.
- The raw output also contains an `ipw` column, and the same scripts can be rerun with `--estimator ipw`.

## Scenario 1 Setup

- `N = 3000`
- `T = 60`
- capacity factors: `[0.90, 0.92, 1.20]`
- forecast gaps: `Delta_control = -0.5`, `Delta_treat = -0.05`
- parameter mode: `moderate_demand_heterogeneous`
- Monte Carlo replications: `R_GTE = 300`, `R_DESIGN = 300`

{markdown_table(selected, 1)}

## Scenario 2 Setup

- `N = 3000`
- `T = 60`
- capacity factors: `[0.85, 1.00, 1.10]`
- forecast-error radii: `eta_control = 30.0`, `eta_treat = 0.2`
- parameter mode: `scenario2_clean_signal`
- Monte Carlo replications: `R_GTE = 300`, `R_DESIGN = 300`

{markdown_table(selected, 2)}

## Validity Checks

Selected-regime numerical validity checks: `{status}`.

The checked rates are the margin, lower-support, and no-overshoot diagnostic violation rates recorded by the simulator. See `theory_region_warnings_combined.txt` for any failures.

## Main Artifacts

- `selected_capacity_regimes_combined.csv`: selected final regimes.
- `summary_scenario1_full_grid.csv`, `summary_scenario2_full_grid.csv`: full-grid summaries.
- `raw_estimates_scenario1_selected.csv`, `raw_estimates_scenario2_selected.csv`: raw estimator draws for selected regimes.
- `figures_abtest_style/`: violin plots in the same style as `abtest/main_abtest_me.py`.
"""
    (output_dir / "controlled_stochastic_final_report.md").write_text(text, encoding="utf-8")


def assemble(scenario1_dir: Path, scenario2_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_source_artifacts(scenario1_dir, output_dir / "scenario1", 1)
    copy_source_artifacts(scenario2_dir, output_dir / "scenario2", 2)

    summary1 = read_required_csv(scenario1_dir / "summary_scenario1_full_grid.csv")
    summary2 = read_required_csv(scenario2_dir / "summary_scenario2_full_grid.csv")
    run1 = read_required_csv(scenario1_dir / "run_level_results_scenario1.csv")
    run2 = read_required_csv(scenario2_dir / "run_level_results_scenario2.csv")

    selected1 = build_selected(summary1, 1)
    selected2 = build_selected(summary2, 2)
    selected = pd.concat([selected1, selected2], ignore_index=True)

    summary1.to_csv(output_dir / "summary_scenario1_full_grid.csv", index=False)
    summary2.to_csv(output_dir / "summary_scenario2_full_grid.csv", index=False)
    selected.to_csv(output_dir / "selected_capacity_regimes_combined.csv", index=False)
    selected.to_csv(output_dir / "selected_capacity_regimes.csv", index=False)
    write_selected_raw(run1, selected, 1, output_dir / "raw_estimates_scenario1_selected.csv")
    write_selected_raw(run2, selected, 2, output_dir / "raw_estimates_scenario2_selected.csv")
    write_warnings(selected, output_dir / "theory_region_warnings_combined.txt")
    write_report(selected, output_dir)
    copy_figures([scenario1_dir, scenario2_dir], output_dir / "figures_abtest_style")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario1-dir", type=Path, required=True)
    parser.add_argument("--scenario2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.scenario1_dir, args.scenario2_dir, args.output_dir)
    print(f"[OK] assembled final outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
