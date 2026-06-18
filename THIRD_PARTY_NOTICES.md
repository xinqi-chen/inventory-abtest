# Third-party Notices

This repository contains original code together with adapted third-party
components used for the trace-driven FreshRetailNet experiments.

The original code in this repository is released under the MIT License. Adapted
third-party components remain under their upstream licenses.

## FreshRetailNet-50K Baseline

- Source: https://github.com/Dingdong-Inc/frn-50k-baseline
- License: Apache-2.0
- License text: `third_party_licenses/Apache-2.0-FreshRetailNet.txt`
- Used in: `trace_driven_freshretailnet/latent_demand_recovery/` and portions
  of `trace_driven_freshretailnet/demand_forecasting/`
- Notes: latent-demand recovery and baseline forecasting components were
  adapted for the inventory A/B-testing experiments in this repository.

Please also cite the FreshRetailNet-50K paper when using the trace-driven
experiments:

```bibtex
@article{2025freshretailnet-50k,
  title={FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail},
  author={Yangyang Wang, Jiawei Gu, Li Long, Xin Li, Li Shen, Zhouyu Fu, Xiangjun Zhou, Xu Jiang},
  year={2025},
  eprint={2505.16319},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.16319}
}
```

## PyPOTS

- Source: https://github.com/WenjieDu/PyPOTS
- License: BSD-3-Clause
- License text: `third_party_licenses/BSD-3-Clause-PyPOTS.txt`
- Copyright: Copyright (c) 2023-present, Wenjie Du
- Used in: `trace_driven_freshretailnet/latent_demand_recovery/pypots/`
- Notes: the vendored/adapted PyPOTS components support the imputation models
  used for latent demand recovery.

## Additional Model References

The trace-driven forecasting code also documents model-level references in
`trace_driven_freshretailnet/README.md`, including TimesNet, DLinear,
Temporal Fusion Transformer / PyTorch Forecasting, and LightGBM.
