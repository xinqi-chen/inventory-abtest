# Paper-to-experiment map

References below follow the current [paper draft](../paper/ABtestInventory_ec26.pdf),
especially Table 1, §5 and Appendix C. For commands, see [reproduction](reproduction.md).
Bias means the average estimated GTE minus the global-treatment/global-control benchmark.

| Paper | Experiment and purpose | Existing entry point / settings | Generated result |
|---|---|---|---|
| §4.1, §5.1, Fig. 2 | Controlled S1: reduce downward mean bias; examine SW negative bias, IR positive bias under binding capacity, and PR bias ≤ IR bias | `controlled_stochastic_simulations/run_final_experiments.sh`; Δ control/treatment = −0.50/−0.05; capacity factors 0.90/0.92/1.20 | `gte_violin_scenario1_*.png`, `summary_scenario1_full_grid.csv` |
| §4.2, §5.1, Fig. 3 | Controlled S2: reduce forecast-error dispersion; examine positive SW/PR bias, IR near zero, and PR close to SW in a large system | Same runner; error radii 30.0/0.2; capacity factors 0.85/1.00/1.10 | `gte_violin_scenario2_*.png`, `summary_scenario2_full_grid.csv` |
| §5.2, Fig. 4 | Trace-driven S1: examine mean-bias and interference mechanisms on recovered retail demand | `trace_driven_freshretailnet/abtest/main_abtest_me.py --s1_mode`; DLinear pair; substitution = 1; capacity factors 0.9/1.2/1.8 | `S1_Dtrue_TimesNet_C_raw_DLinear_T_TimesNet_DLinear_sub_1_cap_<factor>/...` |
| §5.2, Fig. 5 | Trace-driven S2: examine dispersion and carryover mechanisms on recovered retail demand | Same program, `--s2_mode`; Weekday control / TFT treatment; substitution = 1; capacity factors 0.6/0.9/1.8 | `S2_Dtrue_TimesNet_C_TimesNet_Weekday_T_TimesNet_TFT_sub_1_cap_<factor>/...` |
| Table 2 | Characterize the forecast pairs using WAPE (error magnitude) and WPE (signed forecast bias) | Trace-driven GT/GC runs and `scripts/export_abtest_latex.py` | `GT_*/exp_summary.csv`, `GC_*/exp_summary.csv`; exported `*_gt_gc.tex` |

Controlled results are collected under
`controlled_stochastic_simulations/outputs_reproduced/final_theory_verification/`;
plots are in `figures_abtest_style/`. Both scenarios use N = 3000, T = 60 and
300 benchmark/design replications in the final runner.

Running the trace-driven experiments writes full outputs to
`trace_driven_freshretailnet/abtest/abtest_outputs/251221/` locally. This directory
is ignored by Git; only selected paper figures and CSVs are included in
[`results/`](../README.md#results).
Each configuration contains `abtest_summary_*.csv` and `figs/gte_violin_*.png`.
These runs use a 7-day horizon and 30 design replications.
The commands select the current implementations of `s1_mode` and `s2_mode`;
the directory names identify forecast inputs, not the full in-simulator transformation.

The controlled module constructs uniform-demand settings for theory checks;
the trace-driven module examines mechanisms using recovered demand and stockout
substitution. The full S1 runner also includes SSA and TFT pairs; the full S2
runner includes LightGBM and SSA treatments. Those additional pairs and
substitution-disabled runs are outside the configurations of Figs. 4–5.
