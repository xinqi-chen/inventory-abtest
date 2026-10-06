# Reproduction

Commands below use the existing code and defaults. A small controlled simulation,
plotting and assembly workflow has been exercised. The full trace-driven pipeline
and a fresh installation of the exact dependency pins have not been verified.
See the [experiment map](experiment_design.md) for the purpose of each run.

## Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate inventory-abtest
```

The environment specifies Python 3.8. Module requirements are in
`controlled_stochastic_simulations/requirements.txt` and
`trace_driven_freshretailnet/requirements.txt`. Runtime, peak memory and the
hardware used for the paper outputs are not documented. Recovery and DLinear
select CPU/GPU based on availability, but the current TFT training sets `gpus=1`
and prediction calls `.cuda()`: use a CUDA-enabled NVIDIA GPU for the complete
trace-driven workflow. A Mac/CPU-only system cannot execute that TFT path unchanged.

## Controlled experiments

No external dataset is needed. From the repository root:

```bash
cd controlled_stochastic_simulations
bash run_final_experiments.sh
cd ..
```

This runs both scenarios, plots the estimates and assembles the report under
`controlled_stochastic_simulations/outputs_reproduced/final_theory_verification/`.
Look for `figures_abtest_style/`, `summary_scenario{1,2}_full_grid.csv` and
`controlled_stochastic_final_report.md`. The default base seed is `20260614`;
the runner uses 8 workers. Settings are listed in the [module README](../controlled_stochastic_simulations/README.md).
To use a separate output folder: `OUT_ROOT=outputs_new bash run_final_experiments.sh`.

## Trace-driven experiments

Start from the repository root. Keep the working-directory changes below in order.

### 1. Download and recover demand

```bash
cd trace_driven_freshretailnet
python scripts/download_freshretailnet.py
cd latent_demand_recovery/exp
python app.py --model TimesNet
cd ../..
```

The download pins FreshRetailNet revision
`2acbe01460f63fb293f090b6cdaf94ac588ab85c` and saves `frn_50k_local_dataset/`
(about 115 MB compressed download, 2.5 GB local Arrow data).
Recovery creates `latent_demand_recovery/exp/demand/demand_TimesNet.parquet`
and `demand_eval_TimesNet.parquet`; its seed is `1024`.

### 2. Generate forecast inputs for Figs. 4–5

From `trace_driven_freshretailnet/` after step 1:

```bash
cd demand_forecasting/DLinear
bash run_dlinear.sh --input-sources "raw,TimesNet"
cd ../TFT
bash run_tft.sh --input-sources "raw,TimesNet" --prediction-type deterministic
cd ../..
python demand_forecasting/ClassicalModels/run_forecasting.py --recovery_model_name TimesNet
```

The ClassicalModels command must run from `trace_driven_freshretailnet/`;
it generates both Weekday and LightGBM. Each model writes prediction Parquet
files and metric CSVs to `demand_forecasting/<model>/results/`
(Weekday/LightGBM use `ClassicalModels/results/`). Keep the metrics as well as
the predictions: the A/B simulator reads training metrics for forecast corrections.

### 3. Run the paper's trace-driven configurations

From `trace_driven_freshretailnet/`, run all three capacities for each pair:

```bash
for cap in tight medium loose; do
  python abtest/main_abtest_me.py \
    --true_demand_source TimesNet \
    --recovery_control raw --forecast_control DLinear \
    --recovery_treat TimesNet --forecast_treat DLinear \
    --s1_mode --stockout_substitute 1 --capacity_type "$cap" --n_jobs 1
  python abtest/main_abtest_me.py \
    --true_demand_source TimesNet \
    --recovery_control TimesNet --forecast_control Weekday \
    --recovery_treat TimesNet --forecast_treat TFT \
    --s2_mode --stockout_substitute 1 --capacity_type "$cap" --n_jobs 1
done
```

The seed is `251221`. S1 capacity factors are 0.9/1.2/1.8 and S2 factors
are 0.6/0.9/1.8 (tight/medium/loose). Results go to
`abtest/abtest_outputs/251221/<configuration>/<capacity>_<factor>/`, including
`ab_config.json`, `abtest_summary_*.csv`, `figs/gte_violin_*.png` and GT/GC summaries.

### Alternative: run the full existing batches

After step 2, generate SSA inputs before calling the batch runners:

```bash
cd demand_forecasting/SSA
bash run_forecast_1.sh --forecast-models "SSA" --input-sources "raw,TimesNet"
cd ../..
STOCKOUT_SUBSTITUTE=1 bash abtest/run_abtest_s1.sh
STOCKOUT_SUBSTITUTE=1 bash abtest/run_abtest_s2.sh
```

Each batch runs three forecast pairs across three capacities. Defaults are
`MAX_PROCS=4` and `N_JOBS=1`; logs go to `slurm_logs/`.
Use `STOCKOUT_SUBSTITUTE=0` for the additional substitution-disabled runs.

### 4. Export existing outputs

From `trace_driven_freshretailnet/`:

```bash
python scripts/export_abtest_latex.py --seed 251221 --mode S1 --sub 1
python scripts/export_abtest_latex.py --seed 251221 --mode S2 --sub 1
```

Exports go to `outputs/latex/251221/<MODE>/sub_1/{tex,latex_png}/`.
The exporter scans all matching configurations already in the seed folder;
it does not restrict the export to the paper's two forecast pairs.

## Reusing outputs

Skip a stage only when its required outputs are already available at the expected
paths. Recovery needs the local dataset; forecasting needs recovered train/eval
demand; A/B tests need recovered eval demand plus predictions and training metrics;
export needs completed A/B outputs. Full generated data, checkpoints and raw output
directories are ignored by Git. The [paper figures and original summary CSVs](../README.md#results)
are available in a fresh clone; recovery/forecast inputs and full trajectories
needed to rerun the pipeline are not included.

Rerunning recovery, forecasting, A/B tests or export can overwrite files at the
same paths. Preserve outputs you want to compare before rerunning.

## Validation scope and known differences

- Saved results: all 12 figure panels match the paper's decoded images, and the saved GT/GC forecast metrics match Table 2 after rounding. The published CSVs are unchanged copies of the historical outputs.
- Controlled workflow: a temporary-directory smoke run exercised both scenarios at N = 60, T = 8, 3 benchmark replications and 5 design replications, followed by plotting and assembly. All five commands completed and produced six plots. This checks execution, not the statistical conclusions or a full N = 3000 reproduction.
- Trace-driven workflow: the TFT model and vendored PyPOTS imported successfully in an existing Python 3.8 environment. Recovery, model training and the complete A/B batches have not been rerun. That local environment has torchmetrics 1.5.2 and pyarrow 17.0.0, whereas the requirements pin 0.11.4 and 15.0.2; the import check does not certify those pins or a fresh install. PyTorch Forecasting 1.0.0 also brings its own `lightning` dependency; the vendored TFT uses `pytorch_lightning`. Keep the declared versions until a complete environment has been tested.
- DLinear RevIN: the runner passes `--revin False`, but `main.py` uses `type=bool`; argparse therefore interprets the nonempty string as **True**. This differs from the paper appendix's statement that RevIN is disabled. The scripts are preserved; do not interpret this flag as disabling RevIN.
- Forecast corrections: S1/S2 apply training-ME corrections and nonnegative clipping inside the simulator. Current ME multipliers are 0.2 for S1 DLinear, 0.75 for S1 SSA/TFT, and 1.5 for S2 TFT. Forecast-pair names alone do not fully describe the intervention.
- Estimator: the paper's theoretical analysis defines IPW, but the saved figures and documented runs use difference in means under Bernoulli assignment. They need not coincide. The legacy controlled `mean_IPW` summary column contains the selected estimator's mean; use `estimator` and `mean_estimate`.
- Repetition counts: trace GT/GC each run once; SW has 29 valid contrasts from 30 assignments because one assignment contains only controls; IR/PR have 30. Controlled designs use 300 replications per paper capacity.

The saved paper results are a historical reference, not a guarantee that current
training runs reproduce identical numerical outputs. Exact trace-driven
reproduction still requires verification of the training environment, generated
forecast inputs and the full CUDA workflow. No simulation or training code was
changed during the publication review.
