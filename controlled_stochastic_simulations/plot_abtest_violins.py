#!/usr/bin/env python3
"""Plot controlled-stochastic results in the style of abtest/main_abtest_me.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

DISPLAY_LABELS = {"SW": "SW", "IR": "IR", "BR": "PR"}


def plot_one(vals_by_design, labels, gte_true, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=140)
    ax.violinplot(vals_by_design, showmeans=True, showextrema=True, showmedians=False)
    ax.axhline(gte_true, linestyle="--", linewidth=1.5, color="tab:gray")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_plots(output_dir: Path, estimator: str = "diff_in_means") -> None:
    fig_dir = output_dir / "figs_abtest_style"
    fig_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(output_dir / "selected_capacity_regimes.csv")
    design_order = ["SW", "IR", "BR"]

    for scenario in [1, 2]:
        raw_path = output_dir / f"raw_estimates_scenario{scenario}_selected.csv"
        summary_path = output_dir / f"summary_scenario{scenario}_full_grid.csv"
        if not raw_path.exists() or not summary_path.exists():
            continue

        try:
            raw = pd.read_csv(raw_path)
            summary = pd.read_csv(summary_path)
        except EmptyDataError:
            continue
        if raw.empty or summary.empty:
            continue
        raw[estimator] = pd.to_numeric(raw[estimator], errors="coerce")

        for _, pick in selected[selected["scenario"] == scenario].iterrows():
            label = str(pick["label"])
            cf = float(pick["capacity_factor"])
            n = int(pick["N"])

            sub = raw[
                (raw["selected_label"].astype(str) == label)
                & (pd.to_numeric(raw["capacity_factor"], errors="coerce").sub(cf).abs() < 1e-9)
                & (pd.to_numeric(raw["N"], errors="coerce") == n)
            ]
            vals_by_design = []
            labels = []
            for design in design_order:
                vals = sub[sub["design"] == design][estimator].dropna().to_numpy(dtype=float)
                if vals.size > 0:
                    vals_by_design.append(vals)
                    labels.append(DISPLAY_LABELS.get(design, design))
            if not vals_by_design:
                continue

            gte_rows = summary[
                (pd.to_numeric(summary["capacity_factor"], errors="coerce").sub(cf).abs() < 1e-9)
                & (pd.to_numeric(summary["N"], errors="coerce") == n)
            ]
            if gte_rows.empty:
                continue
            gte_true = float(gte_rows.iloc[0]["GTE"])

            out_path = fig_dir / f"gte_violin_scenario{scenario}_{label}_cf{cf:g}_{estimator}.png"
            plot_one(vals_by_design, labels, gte_true, out_path)
            print(f"[OK] {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_rev_selected_demandhet_R300",
    )
    parser.add_argument("--estimator", default="diff_in_means")
    args = parser.parse_args()
    make_plots(args.output_dir, estimator=args.estimator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
