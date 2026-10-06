# Trace-Driven FreshRetailNet Experiments

See the [paper results](../README.md#results) for saved figures and summary CSVs.

This folder contains the FreshRetailNet-50K trace-driven numerical module for inventory A/B testing under shared capacity constraints.

The paper's Fig. 4 uses the S1 DLinear pair; Fig. 5 uses S2 with Weekday
(Naive) control and TFT treatment. Both enable stockout substitution.
See the [paper-to-experiment map](../docs/experiment_design.md) and
[reproduction guide](../docs/reproduction.md#trace-driven-experiments) for
the paper configurations and the full batch workflow.

Latent demand recovery and several baseline forecasting components are adapted
from the FreshRetailNet-50K baseline repository
[frn-50k-baseline](https://github.com/Dingdong-Inc/frn-50k-baseline). See
`../THIRD_PARTY_NOTICES.md` for source and license notices.

---

## Dataset

We use **FreshRetailNet-50K**, a large-scale fresh retail dataset with stockout annotations:

- 50,000 store–SKU time series
- 898 stores across 18 cities in China
- Hourly sales data with stockout censoring

Dataset link:  
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

Download the pinned dataset snapshot and save it in the local folder expected by
the code:

```bash
python scripts/download_freshretailnet.py
```

The command creates:

```text
frn_50k_local_dataset/
```

The Hugging Face download is about 115 MB compressed and the saved local Arrow
dataset is about 2.5 GB. The downloader pins the dataset revision and verifies
the expected `train` and `eval` splits before exiting.

Although the raw data are hourly, all forecasting models and the inventory simulator in this repo operate on **daily aggregated demand** (daily `sale_amount` indexed by `dt`).

---

## Folder Structure

```text
latent_demand_recovery/ # Latent (uncensored) demand recovery
demand_forecasting/ # Demand forecasting models
├── SSA/
├── ClassicalModels/ # Naive, LightGBM
├── DLinear/
└── TFT/
abtest/ # Inventory simulation and A/B testing
├── main_abtest_me.py
├── run_abtest_s1.sh
└── run_abtest_s2.sh
```

---

## Environment Setup

The declared environment uses Python 3.8. The full TFT workflow requires a
CUDA-enabled NVIDIA GPU; CPU fallback is not implemented in its current training
and prediction entry points. See the [validation notes](../docs/reproduction.md#validation-scope-and-known-differences) before treating a run as an exact reproduction.

Environment setup:

```bash
conda create --name py3.8_frn python=3.8
conda activate py3.8_frn
pip install -r requirements.txt
```

Run commands in this README from `trace_driven_freshretailnet/` unless a command changes into a subfolder.

## Experimental Workflow

The pipeline consists of three stages:

1. Latent demand recovery from stockout-censored sales
2. Demand forecasting using recovered and/or raw data
3. Inventory A/B testing under shared capacity constraints

Each stage can be rerun once its required inputs from preceding stages exist.


### Latent Demand Recovery

We recover latent (uncensored) demand from censored sales using imputation models. We use TimesNet in our experiments.
> - TimesNet: [*TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*](https://arxiv.org/abs/2210.02186)  
```bash
cd latent_demand_recovery/exp
python app.py --model TimesNet
cd ../..
```
This produces recovered demand fields (e.g., sale_amount_pred) used as ground truth in trace-driven simulation.


### Demand Forecasting

All forecasting models output point forecasts interpreted as demand mean predictions.

1. SSA (Similar Scenario Average)
SSA: A statistics-based weighted averaging method over historically similar contexts.

```bash
cd demand_forecasting/SSA
bash run_forecast_1.sh --forecast-models "SSA" --input-sources "raw,TimesNet"
cd ../..
```

2. Naive & LightGBM
Naive: A simple baseline that predicts demand at day t using demand at day t-7.

LightGBM: Gradient-boosted decision trees trained on lagged demand features and calendar covariates.
> - Paper: [*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
> - Reference Code link:https://github.com/microsoft/LightGBM

```bash
python demand_forecasting/ClassicalModels/run_forecasting.py --recovery_model_name TimesNet
```

Run this command from `trace_driven_freshretailnet/`; it writes both Weekday
and LightGBM predictions and metrics to `demand_forecasting/ClassicalModels/results/`.

3. DLinear
DLinear: A linear forecasting model with trend–seasonality decomposition.
> - Paper: [*Are Transformers Effective for Time Series Forecasting?*](https://ojs.aaai.org/index.php/AAAI/article/view/26317)  
> - Reference Code link: https://github.com/cure-lab/LTSF-Linear

```bash
cd demand_forecasting/DLinear
bash run_dlinear.sh --input-sources "raw,TimesNet"
cd ../..
```

4. TFT
TFT: A sequence-to-sequence neural forecasting model with attention and static covariates.
> - Paper link: [*Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*](https://arxiv.org/abs/1912.09363)
> - Reference Code link: https://github.com/sktime/pytorch-forecasting

```bash
cd demand_forecasting/TFT
bash run_tft.sh --input-sources "raw,TimesNet" --prediction-type deterministic
cd ../..
```


### Inventory A/B Tests

Inventory A/B tests are implemented in abtest/main_abtest_me.py.
They simulate myopic inventory allocation under a shared capacity constraint and compare treatment vs control forecasts using three experimental designs:

- SW: Switchback randomization
- IR: Item-level randomization
- PR: Pairwise randomization

The full S1 runner requires SSA, DLinear and TFT outputs; the full S2 runner
requires Weekday, LightGBM, SSA and TFT outputs. Complete the forecasting steps
above before running either batch. For only Figs. 4–5, use the direct commands
in the [reproduction guide](../docs/reproduction.md#3-run-the-papers-trace-driven-configurations).

The paper reports only the substitution-enabled DLinear pair in S1 and
Weekday (Naive) → TFT in S2. The batch runners support a broader set:

| Scenario | Control inputs → treatment inputs | Stockout substitution |
| --- | --- | --- |
| S1 | Raw → TimesNet-recovered inputs, using SSA, DLinear or TFT | Off or on |
| S2 | Weekday control → LightGBM, SSA or TFT treatment, all using TimesNet-recovered inputs | Off or on |

Each runner evaluates tight, medium and loose capacity with SW, IR and PR.
Use `STOCKOUT_SUBSTITUTE=0` or `1` with the commands below. The simulator CLI
also accepts control/treatment recovery and forecast selections; the selected
models need their corresponding prediction and training-metric files. These
additional configurations are supported by the code, while `results/` contains
only the figures and CSVs displayed in the paper.

#### Scenario 1

Without stockout substitution:
```bash
bash abtest/run_abtest_s1.sh
```
With stockout substitution:
```bash
STOCKOUT_SUBSTITUTE=1 bash abtest/run_abtest_s1.sh
```

#### Scenario 2

Without stockout substitution:
```bash
bash abtest/run_abtest_s2.sh
```
With stockout substitution:
```bash
STOCKOUT_SUBSTITUTE=1 bash abtest/run_abtest_s2.sh
```



## Export LaTeX Tables and Figures (Aggregation / Post-processing)

Running A/B tests writes full outputs locally to the following per-seed directory.
This directory is ignored by Git and is not included in the repository:

- `./abtest/abtest_outputs/251221/<MODE>_Dtrue_..._sub_<0|1>_cap_<CAP>/...`

This repository includes a post-processing script that scans the seed directory, reads:
- `abtest_summary_*.csv` (for GTE/bias/std tables),
- `figs/gte_violin_*.png` (for violin plots),
- `GT_*/exp_summary.csv` and `GC_*/exp_summary.csv` (for WAPE/WPE tables),

and exports LaTeX-ready tables and figures into a clean output folder:

- `./outputs/latex/251221/<MODE>/sub_<0|1>/tex/`
- `./outputs/latex/251221/<MODE>/sub_<0|1>/latex_png/`

> Note: this script requires that experiment outputs already exist on the machine
> where you run it (e.g., your server). It will not produce results on a fresh
> clone without running experiments.

### Scenario 1 (Bias reduction)

No substitution:
```bash
python scripts/export_abtest_latex.py --seed 251221 --mode S1 --sub 0
```
With substitution:
```bash
python scripts/export_abtest_latex.py --seed 251221 --mode S1 --sub 1
```

### Scenario 2 (Forecast-error dispersion reduction)
No substitution:
```bash
python scripts/export_abtest_latex.py --seed 251221 --mode S2 --sub 0
```
With substitution:
```bash
python scripts/export_abtest_latex.py --seed 251221 --mode S2 --sub 1
```

The exported .tex files are placed under .../tex/, and the corresponding PNGs
are copied into .../latex_png/.
