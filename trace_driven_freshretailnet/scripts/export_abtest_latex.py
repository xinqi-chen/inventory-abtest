#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_abtest_latex.py

Collect outputs from `abtest/main_abtest_me.py` and export LaTeX snippets.

Input layout (produced by main_abtest_me.py):

  <PROJECT_ROOT>/abtest/abtest_outputs/<SEED>/
    <MODEL_SHORTHAND>/
      <CAPACITY_TYPE>_<CAP_MULT>/
        abtest_summary_<MODEL_SHORTHAND>.csv
        figs/
          *.png
        GT_<MODEL_SHORTHAND>/exp_summary.csv
        GC_<MODEL_SHORTHAND>/exp_summary.csv

Output layout (this script):

  <PROJECT_ROOT>/outputs/latex/<SEED>/<MODE>/sub_<0|1>/
    tex/
      <SEED>_<MODE>_sub_<sub>.tex
      <SEED>_<MODE>_sub_<sub>_gt_gc.tex
      <SEED>_<MODE>_sub_<sub>_figures.tex
    latex_png/
      gte_violin_<...>.png

Usage examples:
  python scripts/export_abtest_latex.py --seed 251221 --mode S1 --sub 0
  python scripts/export_abtest_latex.py --seed 251221 --mode S2 --sub 1
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Example model shorthand:
#   S1_Dtrue_TimesNet_C_raw_DLinear_T_TimesNet_DLinear_sub_0_cap_1.8
SECOND_LEVEL_PATTERN = re.compile(
    r"^(?P<mode>S1|S2|S2_ORACLE)"
    r"_Dtrue_(?P<true>[^_]+)"
    r"_C_(?P<rc>[^_]+)_(?P<fc>[^_]+)"
    r"_T_(?P<rt>[^_]+)_(?P<ft>[^_]+)"
    r"_sub_(?P<sub>\d+)"
    r"_cap_(?P<cap>.+)$"
)

# capacity folder example: tight_0.9, medium_1.2, loose_1.8
THIRD_LEVEL_PATTERN = re.compile(r"^(?P<cap_type>loose|medium|tight)_(?P<cap>.+)$")


def escape_tex(s: str) -> str:
    if s is None:
        return ""
    repl = {
        "\\": r"\\",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))


def fmt_num(x: Optional[float], ndigits: int = 2) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return f"{float(x):.{ndigits}f}"


@dataclass(frozen=True)
class PairKey:
    true: str
    rc: str
    fc: str
    rt: str
    ft: str
    cap_str: str  # cap factor embedded in model shorthand


@dataclass
class OneCapResult:
    cap_type: str
    cap_value: float
    summary_csv: Path
    figs_dir: Path
    gt_exp_summary: Optional[Path]
    gc_exp_summary: Optional[Path]


def parse_model_shorthand(name: str) -> Optional[Dict[str, str]]:
    m = SECOND_LEVEL_PATTERN.match(name)
    if not m:
        return None
    d = m.groupdict()
    try:
        float(d["cap"])
    except Exception:
        return None
    return d


def parse_capacity_dir(name: str) -> Optional[Tuple[str, float]]:
    m = THIRD_LEVEL_PATTERN.match(name)
    if not m:
        return None
    cap_type = m.group("cap_type")
    cap_str = m.group("cap")
    try:
        cap_val = float(cap_str)
    except Exception:
        return None
    return cap_type, cap_val


def extract_metrics_from_abtest_summary(csv_path: Path) -> Optional[Dict[str, float]]:
    """
    Expected columns:
      kind, design, gte_true, treat_mean, control_mean, bias, gte_hat_std
    """
    df = pd.read_csv(csv_path)

    gte_mask = (df["kind"] == "GTE_true") & (df["design"] == "GT-GC")
    if not gte_mask.any():
        return None
    gte_row = df.loc[gte_mask].iloc[0]

    out: Dict[str, float] = {
        "GTE": float(gte_row["gte_true"]),
        "GT": float(gte_row["treat_mean"]),
        "GC": float(gte_row["control_mean"]),
    }

    for design in ["SW", "IR", "PR"]:
        m = (df["kind"] == "randomized_design") & (df["design"] == design)
        if not m.any():
            return None
        row = df.loc[m].iloc[0]
        out[f"{design}_bias"] = float(row["bias"])
        out[f"{design}_std"] = float(row["gte_hat_std"])
    return out


def extract_wape_wpe(exp_summary_csv: Path) -> Optional[Tuple[Optional[float], Optional[float]]]:
    try:
        df = pd.read_csv(exp_summary_csv, usecols=["wape_corrected", "wpe_corrected"], nrows=1)
    except Exception:
        return None
    if df.empty:
        return None
    wape = df.iloc[0].get("wape_corrected")
    wpe = df.iloc[0].get("wpe_corrected")
    try:
        wape_v = float(wape)
    except Exception:
        wape_v = None
    try:
        wpe_v = float(wpe)
    except Exception:
        wpe_v = None
    return wape_v, wpe_v


def project_root_from_this_file(this_file: Path) -> Path:
    # scripts/export_abtest_latex.py -> project root is parent of scripts/
    return this_file.resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, required=True, help="Seed folder name under abtest/abtest_outputs/")
    parser.add_argument("--mode", type=str, required=True, choices=["S1", "S2", "S2_ORACLE"], help="Scenario/mode to export")
    parser.add_argument("--sub", type=int, required=True, choices=[0, 1], help="Whether stockout substitution is enabled")
    parser.add_argument(
        "--out_root",
        type=str,
        default="",
        help="Optional override for output root directory. Default: <PROJECT_ROOT>/outputs/latex/",
    )
    args = parser.parse_args()

    project_root = project_root_from_this_file(Path(__file__))
    seed = str(args.seed)
    mode = args.mode
    sub = int(args.sub)

    in_root = project_root / "abtest" / "abtest_outputs" / seed
    if not in_root.exists():
        raise FileNotFoundError(f"Input folder not found: {in_root}")

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else (project_root / "outputs" / "latex")

    out_dir = out_root / seed / mode / f"sub_{sub}"
    tex_dir = out_dir / "tex"
    png_dir = out_dir / "latex_png"
    tex_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    runs: Dict[PairKey, List[OneCapResult]] = defaultdict(list)

    for model_dir in sorted([p for p in in_root.iterdir() if p.is_dir()]):
        md = parse_model_shorthand(model_dir.name)
        if md is None:
            continue
        if md["mode"] != mode:
            continue
        if int(md["sub"]) != sub:
            continue

        pair = PairKey(
            true=md["true"],
            rc=md["rc"],
            fc=md["fc"],
            rt=md["rt"],
            ft=md["ft"],
            cap_str=md["cap"],
        )

        for cap_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            parsed = parse_capacity_dir(cap_dir.name)
            if parsed is None:
                continue
            cap_type, cap_val = parsed

            summary_csv = cap_dir / f"abtest_summary_{model_dir.name}.csv"
            if not summary_csv.exists():
                alts = list(cap_dir.glob("abtest_summary_*.csv"))
                if not alts:
                    continue
                summary_csv = alts[0]

            figs_dir = cap_dir / "figs"
            gt_exp = cap_dir / f"GT_{model_dir.name}" / "exp_summary.csv"
            gc_exp = cap_dir / f"GC_{model_dir.name}" / "exp_summary.csv"

            runs[pair].append(
                OneCapResult(
                    cap_type=cap_type,
                    cap_value=cap_val,
                    summary_csv=summary_csv,
                    figs_dir=figs_dir,
                    gt_exp_summary=gt_exp if gt_exp.exists() else None,
                    gc_exp_summary=gc_exp if gc_exp.exists() else None,
                )
            )

    if not runs:
        raise RuntimeError(
            f"No matching results found under {in_root} for mode={mode}, sub={sub}. "
            f"Did you run abtests and use the same SEED?"
        )

    cap_order = {"tight": 0, "medium": 1, "loose": 2}
    for k in list(runs.keys()):
        runs[k].sort(key=lambda r: (cap_order.get(r.cap_type, 99), r.cap_value))

    # 1) Main table
    table_lines: List[str] = []
    table_lines.append(r"\begin{table}[!ht]")
    table_lines.append(r"\centering")
    table_lines.append(r"\small")
    table_lines.append(r"\begin{tabular}{l l l c c c}")
    table_lines.append(r"\hline")
    table_lines.append(r"true & pair (C$\to$T) & cap & GTE$_{true}$ & SW/IR/PR bias & SW/IR/PR std \\")
    table_lines.append(r"\hline")

    for pair, cap_results in sorted(runs.items(), key=lambda kv: (kv[0].true, kv[0].fc, kv[0].ft, float(kv[0].cap_str))):
        pair_str_tex = escape_tex(f"{pair.rc}-{pair.fc}$\\to${pair.rt}-{pair.ft}")
        true_tex = escape_tex(pair.true)

        for cr in cap_results:
            metrics = extract_metrics_from_abtest_summary(cr.summary_csv)
            if not metrics:
                continue
            bias_str = f"{fmt_num(metrics.get('SW_bias'))}/{fmt_num(metrics.get('IR_bias'))}/{fmt_num(metrics.get('PR_bias'))}"
            std_str = f"{fmt_num(metrics.get('SW_std'))}/{fmt_num(metrics.get('IR_std'))}/{fmt_num(metrics.get('PR_std'))}"
            table_lines.append(
                f"{true_tex} & {pair_str_tex} & {escape_tex(cr.cap_type)} "
                f"& {fmt_num(metrics.get('GTE'))} & {bias_str} & {std_str} \\\\"
            )

    table_lines.append(r"\hline")
    table_lines.append(r"\end{tabular}")
    table_lines.append(r"\caption{A/B-test results (aggregated). Bias/std are reported for (SW/IR/PR).}")
    table_lines.append(r"\label{tab:abtest_results_" + escape_tex(mode.lower()) + f"_sub{sub}" + r"}")
    table_lines.append(r"\end{table}")
    table_tex = "\n".join(table_lines) + "\n"

    # 2) GT/GC forecast accuracy
    gtgc_lines: List[str] = []
    gtgc_lines.append(r"\begin{table}[!ht]")
    gtgc_lines.append(r"\centering")
    gtgc_lines.append(r"\small")
    gtgc_lines.append(r"\begin{tabular}{l l c c}")
    gtgc_lines.append(r"\hline")
    gtgc_lines.append(r"true & pair (C$\to$T) & GT (WAPE/WPE) & GC (WAPE/WPE) \\")
    gtgc_lines.append(r"\hline")

    for pair, cap_results in sorted(runs.items(), key=lambda kv: (kv[0].true, kv[0].fc, kv[0].ft, float(kv[0].cap_str))):
        pair_str_tex = escape_tex(f"{pair.rc}-{pair.fc}$\\to${pair.rt}-{pair.ft}")
        true_tex = escape_tex(pair.true)

        gt_wape = gt_wpe = gc_wape = gc_wpe = None
        for cr in cap_results:
            if cr.gt_exp_summary and cr.gc_exp_summary:
                gt = extract_wape_wpe(cr.gt_exp_summary)
                gc = extract_wape_wpe(cr.gc_exp_summary)
                if gt:
                    gt_wape, gt_wpe = gt
                if gc:
                    gc_wape, gc_wpe = gc
                if gt or gc:
                    break

        gt_cell = f"{fmt_num(gt_wape)}/{fmt_num(gt_wpe)}"
        gc_cell = f"{fmt_num(gc_wape)}/{fmt_num(gc_wpe)}"
        gtgc_lines.append(f"{true_tex} & {pair_str_tex} & {gt_cell} & {gc_cell} \\\\")

    gtgc_lines.append(r"\hline")
    gtgc_lines.append(r"\end{tabular}")
    gtgc_lines.append(r"\caption{Forecast accuracy on the evaluation horizon (corrected WAPE/WPE).}")
    gtgc_lines.append(r"\label{tab:forecast_accuracy_" + escape_tex(mode.lower()) + f"_sub{sub}" + r"}")
    gtgc_lines.append(r"\end{table}")
    gtgc_tex = "\n".join(gtgc_lines) + "\n"

    # 3) Figures
    fig_lines: List[str] = []
    fig_lines.append(r"\begin{figure}[!ht]")
    fig_lines.append(r"\centering")
    fig_lines.append(r"\makebox[0.32\textwidth]{tight}\hfill\makebox[0.32\textwidth]{medium}\hfill\makebox[0.32\textwidth]{loose}\\[0.3em]")

    for pair, cap_results in sorted(runs.items(), key=lambda kv: (kv[0].true, kv[0].fc, kv[0].ft, float(kv[0].cap_str))):
        pair_caption = f"Control: {pair.rc} {pair.fc}, Treatment: {pair.rt} {pair.ft}"
        fig_lines.append(r"\begin{subfigure}{\textwidth}")
        fig_lines.append(r"\centering")

        cap_to_png: Dict[str, Optional[Path]] = {"tight": None, "medium": None, "loose": None}
        for cr in cap_results:
            if not cr.figs_dir.exists():
                continue
            pngs = sorted(cr.figs_dir.glob("*.png"))
            if not pngs:
                continue
            chosen = next((p for p in pngs if "gte_violin" in p.name), pngs[0])

            out_name = f"gte_violin_{pair.true}_{pair.rc}-{pair.fc}_to_{pair.rt}-{pair.ft}_sub{sub}_{cr.cap_type}_{cr.cap_value}.png"
            out_name = re.sub(r"[^A-Za-z0-9._-]+", "_", out_name)
            out_png = png_dir / out_name
            if not out_png.exists():
                shutil.copy2(chosen, out_png)
            cap_to_png[cr.cap_type] = out_png

        for cap_type in ["tight", "medium", "loose"]:
            p = cap_to_png.get(cap_type)
            if p is None:
                fig_lines.append(r"\fbox{\rule{0pt}{1.5in}\rule{0.32\textwidth}{0pt}}")
            else:
                fig_lines.append(rf"\includegraphics[width=0.32\textwidth]{{latex_png/{escape_tex(p.name)}}}")

        fig_lines.append(r"\caption{" + escape_tex(pair_caption) + r"}")
        fig_lines.append(r"\end{subfigure}")

    fig_lines.append(r"\caption{A/B-test GTE violin plots (" + escape_tex(mode) + f", sub={sub}" + r").}")
    fig_lines.append(r"\label{fig:abtest_violin_" + escape_tex(mode.lower()) + f"_sub{sub}" + r"}")
    fig_lines.append(r"\end{figure}")
    figures_tex = "\n".join(fig_lines) + "\n"

    base = f"{seed}_{mode}_sub_{sub}"
    (tex_dir / f"{base}.tex").write_text(table_tex, encoding="utf-8")
    (tex_dir / f"{base}_gt_gc.tex").write_text(gtgc_tex, encoding="utf-8")
    (tex_dir / f"{base}_figures.tex").write_text(figures_tex, encoding="utf-8")

    print(f"[OK] Wrote LaTeX snippets to: {tex_dir}")
    print(f"[OK] Copied figures to:       {png_dir}")


if __name__ == "__main__":
    main()
