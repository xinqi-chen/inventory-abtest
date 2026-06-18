# Inventory A/B Testing Experiments

This repository contains the numerical code for A/B testing of demand-forecasting policies under shared inventory capacity constraints.

It includes two complementary experiment modules:

- `controlled_stochastic_simulations/`: self-contained uniform-demand simulations used to verify the theory.
- `trace_driven_freshretailnet/`: FreshRetailNet-50K trace-driven demand recovery, forecasting, and inventory A/B-test simulations.

## Repository Layout

```text
controlled_stochastic_simulations/
  run_controlled_stochastic.py
  run_final_experiments.sh
  plot_abtest_violins.py
  assemble_final_outputs.py

trace_driven_freshretailnet/
  latent_demand_recovery/
  demand_forecasting/
  abtest/
  scripts/
```

Generated outputs, checkpoints, local dataset files, and cache files are ignored by Git. A fresh clone should download or generate data artifacts locally.

## Environment

For a single environment covering both modules:

```bash
conda env create -f environment.yml
conda activate inventory-abtest
```

Alternatively, install dependencies per module:

```bash
python -m pip install -r controlled_stochastic_simulations/requirements.txt
python -m pip install -r trace_driven_freshretailnet/requirements.txt
```

## Data

The trace-driven experiments use FreshRetailNet-50K:

https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

The local dataset folder is not committed. Download and validate the pinned
FreshRetailNet-50K snapshot with:

```bash
python trace_driven_freshretailnet/scripts/download_freshretailnet.py
```

This saves the dataset in the folder expected by the code:

```text
trace_driven_freshretailnet/frn_50k_local_dataset/
```

The Hugging Face download is about 115 MB compressed and the saved local Arrow
dataset is about 2.5 GB.

The trace-driven pipeline also generates recovered-demand and forecast-result files under ignored output directories.

## Controlled Stochastic Simulations

```bash
cd controlled_stochastic_simulations
bash run_final_experiments.sh
```

See `controlled_stochastic_simulations/README.md` for settings and output locations.

## Trace-Driven FreshRetailNet Experiments

```bash
cd trace_driven_freshretailnet
```

Run the trace-driven workflow in stages:

```bash
cd latent_demand_recovery/exp
python app.py --model TimesNet

cd ../../demand_forecasting/DLinear
bash run_dlinear.sh --input-sources "raw,TimesNet"

cd ../TFT
bash run_tft.sh --input-sources "raw,TimesNet" --prediction-type deterministic

cd ../../
STOCKOUT_SUBSTITUTE=1 bash abtest/run_abtest_s1.sh
STOCKOUT_SUBSTITUTE=1 bash abtest/run_abtest_s2.sh
```

See `trace_driven_freshretailnet/README.md` for the full workflow.

## License and Third-party Code

Original code in this repository is released under the MIT License. The
trace-driven module includes components adapted from third-party projects; see
`THIRD_PARTY_NOTICES.md` for source and license notices.
