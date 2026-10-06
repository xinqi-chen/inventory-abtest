# Inventory A/B Testing Experiments

Code and saved results for [*Experimental Designs for Multi-Item Multi-Period Inventory Control*](paper/ABtestInventory_ec26.pdf).

We study how **shared capacity and inventory carryover bias A/B estimates of inventory-policy improvements**. Forecasting methods construct the treatment/control interventions; the target is the global treatment effect (GTE), the average reward under all-treatment minus all-control.

## Main Findings

We compare switchback (SW), item-level (IR), and independent item–period (PR) randomization in controlled simulations and FreshRetailNet-50K trace-driven experiments.

| Intervention | Main findings in the paper experiments |
| --- | --- |
| S1: reduce downward forecast bias | SW underestimates GTE; IR overestimates it under tight capacity. PR reduces IR's positive bias but can underestimate GTE. |
| S2: reduce forecast-error dispersion | IR stays near GTE; SW and PR overestimate it through inventory carryover. |

These are results for the paper's configurations, not guarantees for every forecast pair or capacity regime. See the [paper-to-experiment map](docs/experiment_design.md) for the theoretical conditions and experiment settings.

## Results

The following paper figures and original summary CSVs are included in `results/`; no experiment run is needed to view them. Code also supports substitution-disabled runs and additional forecast pairs, described in the [trace-driven README](trace_driven_freshretailnet/README.md#inventory-ab-tests).

| Paper result | Images | Original CSVs |
| --- | --- | --- |
| Fig. 2: controlled S1 | [tight](results/controlled/scenario1/figures/gte_violin_scenario1_tight_cf0.9_diff_in_means.png) / [medium](results/controlled/scenario1/figures/gte_violin_scenario1_medium_cf0.92_diff_in_means.png) / [loose](results/controlled/scenario1/figures/gte_violin_scenario1_loose_cf1.2_diff_in_means.png) | [summary](results/controlled/scenario1/summary_scenario1_full_grid.csv) |
| Fig. 3: controlled S2 | [tight](results/controlled/scenario2/figures/gte_violin_scenario2_tight_cf0.85_diff_in_means.png) / [medium](results/controlled/scenario2/figures/gte_violin_scenario2_medium_cf1_diff_in_means.png) / [loose](results/controlled/scenario2/figures/gte_violin_scenario2_loose_cf1.1_diff_in_means.png) | [summary](results/controlled/scenario2/summary_scenario2_full_grid.csv) |
| Fig. 4: FreshRetailNet S1 | [tight](results/freshretailnet/gte_violin_S1_Dtrue_TimesNet_C_raw_DLinear_T_TimesNet_DLinear_sub_1_cap_0.9.png) / [medium](results/freshretailnet/gte_violin_S1_Dtrue_TimesNet_C_raw_DLinear_T_TimesNet_DLinear_sub_1_cap_1.2.png) / [loose](results/freshretailnet/gte_violin_S1_Dtrue_TimesNet_C_raw_DLinear_T_TimesNet_DLinear_sub_1_cap_1.8.png) | [tight](results/freshretailnet/abtest_summary_S1_Dtrue_TimesNet_C_raw_DLinear_T_TimesNet_DLinear_sub_1_cap_0.9.csv) / [medium](results/freshretailnet/abtest_summary_S1_Dtrue_TimesNet_C_raw_DLinear_T_TimesNet_DLinear_sub_1_cap_1.2.csv) / [loose](results/freshretailnet/abtest_summary_S1_Dtrue_TimesNet_C_raw_DLinear_T_TimesNet_DLinear_sub_1_cap_1.8.csv) |
| Fig. 5: FreshRetailNet S2 | [tight](results/freshretailnet/gte_violin_S2_Dtrue_TimesNet_C_TimesNet_Weekday_T_TimesNet_TFT_sub_1_cap_0.6.png) / [medium](results/freshretailnet/gte_violin_S2_Dtrue_TimesNet_C_TimesNet_Weekday_T_TimesNet_TFT_sub_1_cap_0.9.png) / [loose](results/freshretailnet/gte_violin_S2_Dtrue_TimesNet_C_TimesNet_Weekday_T_TimesNet_TFT_sub_1_cap_1.8.png) | [tight](results/freshretailnet/abtest_summary_S2_Dtrue_TimesNet_C_TimesNet_Weekday_T_TimesNet_TFT_sub_1_cap_0.6.csv) / [medium](results/freshretailnet/abtest_summary_S2_Dtrue_TimesNet_C_TimesNet_Weekday_T_TimesNet_TFT_sub_1_cap_0.9.csv) / [loose](results/freshretailnet/abtest_summary_S2_Dtrue_TimesNet_C_TimesNet_Weekday_T_TimesNet_TFT_sub_1_cap_1.8.csv) |

Table 2 forecast metrics: S1 [treatment](results/freshretailnet/exp_summary_S1_GT.csv) / [control](results/freshretailnet/exp_summary_S1_GC.csv); S2 [treatment](results/freshretailnet/exp_summary_S2_GT.csv) / [control](results/freshretailnet/exp_summary_S2_GC.csv).

Saved numerical estimates use difference in means; the theoretical analysis defines IPW. `BR` in controlled CSVs is the displayed `PR`; `Weekday` is the paper's Naive benchmark.

## Reproduce

```bash
conda env create -f environment.yml
conda activate inventory-abtest
cd controlled_stochastic_simulations
bash run_final_experiments.sh
```

- [Controlled simulations](controlled_stochastic_simulations/README.md): self-contained, with no external dataset.
- [Trace-driven experiments](docs/reproduction.md#trace-driven-experiments): download → demand recovery → forecasting → A/B tests → export. The current TFT workflow requires a CUDA GPU.
- [Reproduction notes and validation scope](docs/reproduction.md#validation-scope-and-known-differences): saved paper outputs are verified; the full trace-driven pipeline has not been validated in a fresh environment.

Only selected paper figures and CSVs are committed. Full experiment outputs, datasets, forecasts, checkpoints and logs are generated locally and ignored by Git.

## Citation and License

Paper authors: Xinqi Chen, Xingyu Bai, Zeyu Zheng and Nian Si. Citation metadata: [CITATION.cff](CITATION.cff).

Original code: [MIT](LICENSE). Adapted components retain their upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
