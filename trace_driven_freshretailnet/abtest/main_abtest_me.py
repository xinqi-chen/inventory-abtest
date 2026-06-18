from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace, skewnorm, nbinom
from datasets import load_from_disk
import pickle
# from probabilistic_evaluation import evaluate_probabilistic_forecast, compute_and_plot_pit

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# --------------------
# Global config and paths
# --------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Unique item identifier
ITEM_KEY = 'item_key'

# A/B test start time index (relative to dataset)
TEST_START_T_IDX = 90

# region design matrix and substitution matrices
def gen_assignment_matrix(N: int, T: int, type: str, p_model2: float = 0.5, seed: int = 2027) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if type == 'IR':
        W = rng.binomial(1, p_model2, size=(N, 1)) * np.ones((1, T))
    elif type == 'SW':
        W = rng.binomial(1, p_model2, size=(1, T)) * np.ones((N, 1))
    elif type == 'PR':
        W = rng.binomial(1, p_model2, size=(N, T))
    elif type == 'GT':
        W = np.ones((N, T))
    elif type == 'GC':
        W = np.zeros((N, T))
    else:
        raise ValueError("Invalid type. Choose from 'SW','IR','PR','GT','GC'.")
    return W.astype(np.int8)

def gen_store_transition_mats(items_df: pd.DataFrame, seed: int = 2027) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    P_by_store = {}
    for sid, grp in items_df.groupby('store_id'):
        m = len(grp)
        if m <= 1:
            P_by_store[int(sid)] = np.zeros((m, m), float)
            continue
        P = np.zeros((m, m), float)
        for i in range(m):
            row = rng.dirichlet(np.ones(m - 1))
            k = 0
            for j in range(m):
                if j == i: continue
                P[i, j] = row[k]; k += 1
        P_by_store[int(sid)] = P
    return P_by_store

def gen_hierarchical_store_transition_mats(items_df: pd.DataFrame, seed: int = 2027) -> Dict[int, np.ndarray]:
    """
    Build stockout substitution matrices using product hierarchy similarity.
    Similarity priority: management > first > second > third.
    """
    rng = np.random.default_rng(seed)
    P_by_store = {}

    # Define hierarchy levels and weights (higher weight = more important)
    hierarchy_levels = [
        'management_group_id', 
        'first_category_id', 
        'second_category_id', 
        'third_category_id'
    ]
    # Exponentially increasing weights
    level_weights = {'third_category_id': 8, 'second_category_id': 4, 'first_category_id': 2, 'management_group_id': 1}

    # Ensure required columns exist
    for col in hierarchy_levels:
        if col not in items_df.columns:
            raise ValueError(f"Missing required hierarchy column in items_df: {col}")

    for sid, grp in items_df.groupby('store_id'):
        m = len(grp)
        if m <= 1:
            P_by_store[int(sid)] = np.zeros((m, m), float)
            continue

        P = np.zeros((m, m), float)
        grp_reset = grp.reset_index(drop=True)

        for i in range(m):
            scores = np.zeros(m, dtype=float)
            item_i = grp_reset.loc[i]
            
            for j in range(m):
                if i == j: continue
                item_j = grp_reset.loc[j]
                
                # Compute similarity score
                score = 0
                for level in hierarchy_levels:
                    if item_i[level] == item_j[level]:
                        score += level_weights[level]
                scores[j] = score

            # Softmax to probabilities
            scores[i] = -np.inf  # ensure self-probability is zero
            exp_scores = np.exp(scores - np.max(scores))  # stability
            probabilities = exp_scores / np.sum(exp_scores)
            P[i, :] = probabilities

        P_by_store[int(sid)] = P
    return P_by_store

# endregion


# region demand predictor
# ---------- NB2: vectorized quantiles ----------
def _nb2_to_rp(mu: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = 1.0 / np.clip(alpha, 1e-12, None)
    p = r / (r + np.clip(mu, 1e-12, None))
    return r, p

def _calc_wape_wpe_me(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Lightweight WAPE/WPE/ME calculator.
    WAPE = sum(|pred - true|) / sum(true)
    WPE  = sum(pred - true) / sum(true)
    """
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if not np.any(mask):
        return {"WAPE": float("nan"), "WPE": float("nan"), "ME": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    denom = np.sum(yt)
    if denom == 0:
        return {
            "WAPE": 0.0 if np.sum(np.abs(yp)) == 0 else float("inf"),
            "WPE": np.sum(yp - yt) / 1e-9,
            "ME": float(np.mean(yp - yt))
        }
    wape = np.sum(np.abs(yp - yt)) / denom
    wpe = np.sum(yp - yt) / denom
    me = float(np.mean(yp - yt))
    return {"WAPE": float(wape), "WPE": float(wpe), "ME": me}

def nb_ppf_vec(q: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    r, p = _nb2_to_rp(mu, alpha)
    return nbinom.ppf(q, r, p).astype(float)


class Predictor:
    """
    Unified interface to load prediction results (mean and distribution params).
    - SSA/DLinear: load mu/sigma assuming normal.
    - TFT: load all quantile columns.
    """
    def __init__(self, recovery_model: str, forecast_model: str):
        self.recovery_model = recovery_model
        self.forecast_model = forecast_model

        pred_path = self._get_prediction_path()
        print(f"Get prediction results path: {pred_path}")
        df_pred = pd.read_parquet(pred_path)

        # Normalize 'dt' column to 'YYYY-MM-DD' string format
        dt_col = 'dt' if 'dt' in df_pred.columns else 'date'
        if not pd.api.types.is_string_dtype(df_pred[dt_col]):
            df_pred[dt_col] = pd.to_datetime(df_pred[dt_col]).dt.strftime('%Y-%m-%d')
        if dt_col != 'dt':
            df_pred = df_pred.rename(columns={dt_col: 'dt'})

        df_pred[ITEM_KEY] = df_pred['store_id'].astype(str) + '_' + df_pred['product_id'].astype(str)

        # Build lookup by model type
        if self.forecast_model in ['SSA', 'DLinear']:
            self.dist_type = 'normal'
            #  Correctly identify the point forecast column ---
            # SSA and DLinear both use 'sale_amount_pred' as the point forecast column name
            # in their output parquet files.
            mu_col = 'sale_amount_pred' 
            sigma_col = 'pred_sigma'
            if mu_col not in df_pred.columns:
                raise ValueError(f"Column '{mu_col}' not found in prediction file for {self.forecast_model}")
            if sigma_col not in df_pred.columns:
                raise ValueError(f"Column '{sigma_col}' not found in prediction file for {self.forecast_model}")

            df_pred = df_pred.rename(columns={mu_col: 'mu', sigma_col: 'sigma'})
            self._pred_lookup = df_pred.set_index([ITEM_KEY, 'dt'])[['mu', 'sigma']]
        
        elif self.forecast_model in ['LightGBM', 'Weekday', 'SeasonalNaive']:
            # These models provide a point forecast without a sigma.
            # We'll treat them as a normal distribution with sigma=0 for optimization.
            self.dist_type = 'normal'
            mu_col = 'sale_amount_pred'
            if mu_col not in df_pred.columns:
                raise ValueError(f"Column '{mu_col}' not found in prediction file for {self.forecast_model}")
            df_pred = df_pred.rename(columns={mu_col: 'mu'})
            df_pred['sigma'] = 0.0 # Assume zero variance for deterministic forecast
            self._pred_lookup = df_pred.set_index([ITEM_KEY, 'dt'])[['mu', 'sigma']]

        elif self.forecast_model == 'TFT':
            quantile_cols = [col for col in df_pred.columns if col.startswith('q_')]
            #  Handle both deterministic and quantile TFT outputs ---
            if quantile_cols:
                # Quantile prediction mode
                self.dist_type = 'quantiles'
                # Convert quantile names to floats and sort
                self.quantile_levels = sorted([float(q.split('_')[1]) for q in quantile_cols])
                # Keep columns ordered by levels
                self.quantile_cols_sorted = [f"q_{q:.2f}" for q in self.quantile_levels]
                self._pred_lookup = df_pred.set_index([ITEM_KEY, 'dt'])[self.quantile_cols_sorted]
            elif 'sale_amount_pred' in df_pred.columns:
                # Deterministic prediction mode
                self.dist_type = 'normal' # Treat as normal with sigma=0 for optimization logic
                df_pred = df_pred.rename(columns={'sale_amount_pred': 'mu'})
                df_pred['sigma'] = 0.0 # Assume zero variance for deterministic forecast
                self._pred_lookup = df_pred.set_index([ITEM_KEY, 'dt'])[['mu', 'sigma']]
            else:
                raise ValueError("TFT prediction file must contain either quantile columns (q_*) or 'sale_amount_pred' column.")

        else:
            raise ValueError(f"Unsupported forecast model: {self.forecast_model}")

        print(f"✅ Predictor for '{self.forecast_model}' on '{self.recovery_model}' initialized. Distribution type: '{self.dist_type}'.")

    def _get_prediction_path(self) -> Path:
        """Build prediction file path from model combo."""
        base_path = PROJECT_ROOT / 'demand_forecasting'
        
        if self.forecast_model in ['LightGBM', 'Weekday']:
            # LightGBM and Weekday results are in the ClassicalModels directory
            # The user specified that recovery_model will not be 'raw' for these.
            # Filename is like: LightGBM_predictions_recovered_by_TimesNet.parquet
            filename = f"{self.forecast_model}_predictions_recovered_by_{self.recovery_model}.parquet"
            return base_path / 'ClassicalModels' / 'results' / filename

        elif self.forecast_model in ['SSA', 'SeasonalNaive']:
            if self.recovery_model == 'raw':
                filename = f"{self.forecast_model}_predictions_censored.parquet"
            else:
                filename = f"{self.forecast_model}_predictions_recovered_by_{self.recovery_model}.parquet"
            return base_path / 'SSA' / 'results' / filename # Both are in the SSA directory


        elif self.forecast_model == 'DLinear':
            #  Update DLinear file loading logic ---
            # The previous logic searched for a subdirectory based on 'run_desc'.
            # This is updated to directly load the .parquet file from the 'results'
            # directory, matching the new, simplified saving logic in exp_main.py.
            if self.recovery_model == 'raw':
                filename = "DLinear_predictions_censored.parquet"
            else:
                filename = f"DLinear_predictions_recovered_by_{self.recovery_model}.parquet"
            results_dir = base_path / 'DLinear' / 'results'
            return results_dir / filename

        elif self.forecast_model == 'TFT':
            #  Update TFT file loading logic to use uppercase 'TFT' ---
            # The previous filename for raw input was 'tft_predictions_censored.parquet'.
            # This is updated to 'TFT_predictions_censored.parquet' to match the new saving logic.
            if self.recovery_model == 'raw':
                filename = "TFT_predictions_censored.parquet"
            else:
                filename = f"TFT_predictions_recovered_by_{self.recovery_model}.parquet"
            results_dir = base_path / 'TFT' / 'results'
            return results_dir / filename
        else:
            raise ValueError(f"Unknown forecast model: {self.forecast_model}")

    def get_predictions_for_slice(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Return predictions for a given data slice.
        Aligns index with df_slice and returns the distribution columns needed:
        - normal: 'mu', 'sigma', 'dist_type'
        - quantiles: 'q_0.05', ..., 'dist_type'
        """
        idx = pd.MultiIndex.from_frame(df_slice[[ITEM_KEY, 'dt']])
        pred_df = self._pred_lookup.reindex(idx).fillna(0)
        pred_df = pred_df.reset_index()
        pred_df = pred_df.rename(columns={"level_0": ITEM_KEY, "level_1": "dt"})
        # Attach identifiers to keep alignment by store/product
        pred_df[['store_id', 'product_id']] = df_slice[['store_id', 'product_id']].to_numpy()
        pred_df['dist_type'] = self.dist_type
        pred_df.index = df_slice.index
        return pred_df


def _align_pred_info_to_slice(pred_info_all: pd.DataFrame, df_slice: pd.DataFrame) -> pd.DataFrame:
    """Align a precomputed prediction dataframe to a data slice by keys, not by row order.

    This avoids relying on matching positional indices between `work` and `pred_info_all`.
    The returned dataframe has the same row order and index as `df_slice`.
    """
    if pred_info_all is None:
        raise ValueError("pred_info_all is None")
    # Keys that uniquely identify a row at a given time.
    key_cols = ['store_id', 'product_id', 'dt']
    missing = [c for c in key_cols if c not in df_slice.columns]
    if missing:
        raise KeyError(f"df_slice missing key columns: {missing}")
    pred = pred_info_all
    # Fast filter by dt if possible (usually `dt` is constant within df_slice).
    if 'dt' in pred.columns and df_slice['dt'].nunique() == 1:
        pred = pred[pred['dt'] == df_slice['dt'].iloc[0]]
    # De-dup in case upstream saved duplicates.
    if all(c in pred.columns for c in key_cols):
        pred = pred.drop_duplicates(subset=key_cols, keep='first')
    aligned = df_slice[key_cols].merge(pred, on=key_cols, how='left', sort=False)
    aligned.index = df_slice.index
    return aligned


def _precompute_predictions_once(work: pd.DataFrame,
                                 items_index: pd.DataFrame,
                                 rows_by_t: Dict[int, np.ndarray],
                                 test_times: np.ndarray,
                                 pred_ctrl,
                                 pred_trt,
                                 cfg2: dict) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    work_idxed = work.merge(items_index, on=['store_id','product_id'], how='left', sort=False)
    order_by_t = {t: work_idxed.iloc[rows_by_t[t]]['item_idx'].to_numpy().argsort() for t in test_times}
    rows_sorted_by_t = {t: rows_by_t[t][order_by_t[t]] for t in test_times}

    pred_info_ctrl_all = pred_ctrl.get_predictions_for_slice(work)
    if pred_trt:
        pred_info_trt_all = pred_trt.get_predictions_for_slice(work)
    else:
        pred_info_trt_all = None # Not used in oracle mode

    need = {'store_id','product_id','dt'}
    assert need.issubset(pred_info_ctrl_all.columns), pred_info_ctrl_all.columns
    return rows_sorted_by_t, pred_info_ctrl_all, pred_info_trt_all

def _me_scale_for_mode(forecast_model: str, cfg2: dict) -> float:
    """
    Return ME scaling factor based on forecast model and mode.
    - s1_mode: DLinear -> 0.2 ; SSA/TFT -> 0.75 ; others -> 1.0
    - s2_mode: TFT -> 1.5 ; others -> 1.0 (unless overridden by s1_mode)
    - default: DLinear -> 0.5 ; others -> 1.0
    """
    if cfg2.get('s1_mode', False):
        if forecast_model == 'DLinear':
            return 0.2
        if forecast_model in ['SSA', 'TFT']:
            return 0.75
        return 1.0
    if cfg2.get('s2_mode', False) and forecast_model == 'TFT':
        return 1.5
    if forecast_model == 'DLinear':
        return 0.5
    return 1.0


def _load_group_wpe_lookup(forecast_model: str, recovery_model: str, true_demand_source: str) -> Optional[Dict[str, Dict]]:
    """
    Load grouped ME table for LightGBM. Returns mapping: {group_type: {group_value: me}}.
    """
    if forecast_model != 'LightGBM':
        return None

    run_type = 'censored' if recovery_model == 'raw' else 'recovered'
    metrics_filename = f"{forecast_model}_train_group_metrics_{run_type}_by_{true_demand_source}.csv"

    if forecast_model in ['LightGBM', 'Weekday']:
        model_dir = 'ClassicalModels'
    elif forecast_model in ['SSA', 'SeasonalNaive']:
        model_dir = 'SSA'
    else:
        model_dir = forecast_model
    metrics_path = PROJECT_ROOT / 'demand_forecasting' / model_dir / 'results' / metrics_filename

    try:
        df = pd.read_csv(metrics_path)
        by_type = {}
        for g, sub in df.groupby('group_type'):
            # Prefer ME column; fall back to lowercase if needed.
            if 'ME' in sub.columns:
                me_col = 'ME'
            elif 'me' in sub.columns:
                me_col = 'me'
            else:
                raise KeyError("Grouped metrics file missing ME/me column.")
            by_type[g] = dict(zip(sub['group_value'], sub[me_col]))
        print(f"✅ Loaded grouped ME ({len(df)} rows) from '{metrics_path.name}'.")
        return by_type
    except FileNotFoundError:
        print(f"⚠️ Grouped ME file not found: '{metrics_path.name}'. Fallback to global ME.")
        return None
    except Exception as e:
        print(f"⚠️ Failed to load grouped ME from '{metrics_path.name}': {type(e).__name__}: {e}")
        return None

def _get_control_wpe(cfg2: dict) -> Dict:
    """Load control-group ME from train_metrics.csv based on config."""
    forecast_model = cfg2['forecast_control']
    recovery_model = cfg2['recovery_control']
    true_demand_source = cfg2['true_demand_source']

    run_type = 'censored' if recovery_model == 'raw' else 'recovered'

    metrics_filename = f"{forecast_model}_train_metrics_{run_type}_by_{true_demand_source}.csv"

    #  Correct path for SSA and SeasonalNaive models ---
    # Both SSA and SeasonalNaive models store their results in the 'SSA/results' directory.
    if forecast_model in ['LightGBM', 'Weekday']:
        model_dir = 'ClassicalModels'
    elif forecast_model in ['SSA', 'SeasonalNaive']:
        model_dir = 'SSA'
    else:
        model_dir = forecast_model
    metrics_path = PROJECT_ROOT / 'demand_forecasting' / model_dir / 'results' / metrics_filename
    try:
        metrics_df = pd.read_csv(metrics_path)
        #  Robustly read ME/me column ---
        # Check for lowercase 'me' first, then uppercase 'ME'.
        if 'me' in metrics_df.columns:
            me_col = 'me'
        elif 'ME' in metrics_df.columns:
            me_col = 'ME'
        else:
            raise KeyError("Neither 'me' nor 'ME' column found in metrics file.")
        me_val = metrics_df.loc[metrics_df['subset'] == 'All Eval', me_col].iloc[0]
        scale = _me_scale_for_mode(forecast_model, cfg2)
        me_val *= scale
        group_lookup = _load_group_wpe_lookup(forecast_model, recovery_model, true_demand_source)
        # apply same scaling to grouped values
        if group_lookup is not None and scale != 1.0:
            for g, m in group_lookup.items():
                group_lookup[g] = {k: v * scale for k, v in m.items()}
        print(f"✅ Successfully loaded ME={me_val:.4f} from '{metrics_path.name}' for control group.")
        return {"global": float(me_val), "group_lookup": group_lookup}
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"❌ CRITICAL ERROR in s1_mode: Could not load ME from '{metrics_path}'. Error: {type(e).__name__}: {e}")
        raise

def _get_treatment_wpe(cfg2: dict) -> Dict:
    """Load treatment-group ME from train_metrics.csv for s2 control-side use."""
    forecast_model = cfg2['forecast_treat']
    recovery_model = cfg2['recovery_treat']
    true_demand_source = cfg2['true_demand_source']

    run_type = 'censored' if recovery_model == 'raw' else 'recovered'

    metrics_filename = f"{forecast_model}_train_metrics_{run_type}_by_{true_demand_source}.csv"

    #  Correct path for SSA and SeasonalNaive models ---
    # Both SSA and SeasonalNaive models store their results in the 'SSA/results' directory.
    if forecast_model in ['LightGBM', 'Weekday']:
        model_dir = 'ClassicalModels'
    elif forecast_model in ['SSA', 'SeasonalNaive']:
        model_dir = 'SSA'
    else:
        model_dir = forecast_model
    metrics_path = PROJECT_ROOT / 'demand_forecasting' / model_dir / 'results' / metrics_filename
    try:
        metrics_df = pd.read_csv(metrics_path)
        #  Robustly read ME/me column ---
        # Check for lowercase 'me' first, then uppercase 'ME'.
        if 'me' in metrics_df.columns:
            me_col = 'me'
        elif 'ME' in metrics_df.columns:
            me_col = 'ME'
        else:
            raise KeyError("Neither 'me' nor 'ME' column found in metrics file.")
        me_val = metrics_df.loc[metrics_df['subset'] == 'All Eval', me_col].iloc[0]
        scale = _me_scale_for_mode(forecast_model, cfg2)
        me_val *= scale
        group_lookup = _load_group_wpe_lookup(forecast_model, recovery_model, true_demand_source)
        if group_lookup is not None and scale != 1.0:
            for g, m in group_lookup.items():
                group_lookup[g] = {k: v * scale for k, v in m.items()}
        print(f"✅ Successfully loaded ME={me_val:.4f} from '{metrics_path.name}' for treatment group.")
        return {"global": float(me_val), "group_lookup": group_lookup}
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"❌ CRITICAL ERROR in s2_oracle_mode: Could not load ME from '{metrics_path}'. Error: {type(e).__name__}: {e}")
        raise

def _resolve_group_wpe_vector(wpe_info: Optional[Dict], df_slice: pd.DataFrame,
                              priority=('store_id','city_id','first_category_id','second_category_id','dayofweek')) -> np.ndarray:
    """
    Return an ME (bias) vector aligned to df_slice following priority:
    store_id > city_id > first_category_id > second_category_id > dayofweek.
    Falls back to global ME when no group match is found.
    """
    if wpe_info is None:
        return np.zeros(len(df_slice), dtype=float)
    base = np.full(len(df_slice), float(wpe_info.get('global', 0.0)), dtype=float)
    lookup = wpe_info.get('group_lookup') or {}
    for g in priority:
        if g not in df_slice.columns:
            continue
        mapping = lookup.get(g)
        if not mapping:
            continue
        vals = df_slice[g].tolist()
        for i, v in enumerate(vals):
            if v in mapping:
                base[i] = float(mapping[v])
    return base
# endregion

# region experiment
# ---------- Single design ----------
@dataclass
class OneDesignResult:
    exp_summary: pd.DataFrame
    detail_logs: pd.DataFrame
    out_dir: Path


def run_one_design(design: str, cfg2: dict, capacity_df: pd.DataFrame, price_enhanced_eval_df: pd.DataFrame) -> OneDesignResult:
    from pathlib import Path
    import math, json
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed

    model_str = cfg2['model_shorthand']
    # model_str = f"T_{cfg2['recovery_treat']}_{cfg2['forecast_treat']}_vs_C_{cfg2['recovery_control']}_{cfg2['forecast_control']}"
    out_dir = cfg2['path'] / f"{design}_{model_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create detail directory so worker processes can write into it.
    exp_detail_dir = out_dir / "details_by_exp"
    exp_detail_dir.mkdir(exist_ok=True)

    # ---------- 1) Prepare test panel ----------
    base_work = price_enhanced_eval_df.copy() # noqa
    base_work[ITEM_KEY] = base_work['store_id'].astype(str) + '_' + base_work['product_id'].astype(str)

    # --- Load and define simulated "true" demand (D_true) ---
    true_demand_source = cfg2.get('true_demand_source', 'DLinear') # Default to DLinear if not specified
    if true_demand_source == 'raw':
        # Use original (truncated-to-16h) sale_amount as true demand
        base_work['sale_amount_pred'] = base_work['sale_amount']
        print(f"✅ Set true demand (D_true) to 'raw' (original sale_amount).")
    else:
        try: # noqa
            recovered_demand_path = PROJECT_ROOT / 'latent_demand_recovery' / 'exp' / 'demand' / f'demand_eval_{true_demand_source}.parquet'
            recovered_df = pd.read_parquet(recovered_demand_path)

            # Ensure consistent dt formatting before merging
            base_work['dt'] = pd.to_datetime(base_work['dt']).dt.strftime('%Y-%m-%d')
            recovered_df['dt'] = pd.to_datetime(recovered_df['dt']).dt.strftime('%Y-%m-%d')

            base_work = base_work.merge(
                recovered_df[['store_id', 'product_id', 'dt', 'sale_amount_pred']],
                on=['store_id', 'product_id', 'dt'], how='left')
            print(f"✅ Successfully merged recovered demand ('sale_amount_pred') from '{true_demand_source}' as D_true.")
        except FileNotFoundError:
            print(f"CRITICAL WARNING: Recovered demand file for true_demand_source '{true_demand_source}' not found. 'sale_amount_pred' will be NaN. Simulation will fail.")
            base_work['sale_amount_pred'] = np.nan

    # Precompute weekday for grouped WPE
    base_work['dayofweek'] = pd.to_datetime(base_work['dt']).dt.dayofweek.astype(int)

    base_work['time_idx'] = (pd.to_datetime(base_work['dt']) - pd.to_datetime(base_work['dt']).min()).dt.days + TEST_START_T_IDX
    base_work.sort_values(['store_id','product_id','dt'], inplace=True, ignore_index=True)

    items_index = base_work[['store_id','product_id']].drop_duplicates().reset_index(drop=True)
    items_index['item_idx'] = np.arange(len(items_index), dtype=int)
    N = len(items_index)

    test_times = np.sort(base_work['time_idx'].unique())
    T = len(test_times)

    rows_by_t = {int(t): base_work.index[base_work['time_idx'] == t].to_numpy() for t in test_times}

    base_work['__ukey__'] = (
        base_work['store_id'].astype(str) + '_' +
        base_work['product_id'].astype(str) + '_' +
        base_work['time_idx'].astype(str)
    )
    key_to_rows_by_t = {}
    for t in test_times:
        df_t = base_work.loc[rows_by_t[t]]
        key_to_rows_by_t[t] = dict(zip(df_t['__ukey__'].tolist(), df_t.index.tolist()))

    # store capacity
    cap_map = {int(r.store_id): float(r.capacity) for _, r in capacity_df.iterrows()}

    pred_ctrl = Predictor(
        recovery_model=cfg2['recovery_control'],
        forecast_model=cfg2['forecast_control']
    )

    #  Skip loading treatment predictor in Oracle mode ---
    # In oracle mode, the treatment group's decisions are based on D_true,
    # so we don't need to load its forecast model results.
    if cfg2.get('oracle_mode', False) or cfg2.get('s1_oracle_mode', False):
        pred_trt = None
    else:
        pred_trt = Predictor(
            recovery_model=cfg2['recovery_treat'],
            forecast_model=cfg2['forecast_treat']
        )
    # Substitution matrix by store (using hierarchical similarity)
    # Only build when stockout substitution is enabled to avoid extra work.
    hierarchy_cols = ['store_id', 'product_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']
    items_for_subst = base_work[hierarchy_cols].drop_duplicates()
    if int(cfg2.get('stockout_substitute', 0)) != 0:
        P_by_store = gen_hierarchical_store_transition_mats(items_for_subst, seed=cfg2['seed'])
    else:
        P_by_store = {}

    # ---------- 2) Precompute predictions ----------
    rows_sorted_by_t, pred_info_ctrl_all, pred_info_trt_all = _precompute_predictions_once(
        base_work, items_index, rows_by_t, test_times, pred_ctrl, pred_trt, cfg2
    )

    # ---------- 3) Run experiments (parallel) ----------
    num = int(cfg2['num_experiments']) if design in ('SW','IR','PR') else 1
    n_jobs = int(cfg2.get('n_jobs', -1))  # -1 use all cores; 1 runs serially

    readonly_pack = {
        "items_index": items_index,
        "test_times": test_times,
        "cap_map": cap_map,
        "P_by_store": P_by_store,
        "rows_sorted_by_t": rows_sorted_by_t,
        "pred_info_ctrl_all": pred_info_ctrl_all,
        "pred_info_trt_all": pred_info_trt_all,
        "work_view": base_work[['store_id','product_id','time_idx','dt', ITEM_KEY,
                                'city_id','first_category_id','second_category_id','dayofweek',
                                'actual_sale_price','ordering_cost','holding_cost','sale_amount', 'sale_amount_pred']].copy(),
        "design": design,
        "cfg2": cfg2,
    }

    def _run_single_experiment(exp_id: int, pack: dict):
        print(f"Running: exp_id = {exp_id}")
        work = pack["work_view"].copy()
        work['inventory'] = 0.0
        transfer_ratios = []  # incoming / final demand per item-time (skip if final demand == 0)
        overshoot_flags = []  # 1 if I >= S (order qty = 0), else 0
        design_local = pack["design"]
        is_s1 = bool(pack["cfg2"].get('s1_mode', False) or pack["cfg2"].get('s1_oracle_mode', False))
        is_s2 = bool(pack["cfg2"].get('s2_mode', False) or pack["cfg2"].get('s2_oracle_mode', False))
        # Collect corrected-forecast metrics for all GT / GC experiments.
        collect_metrics = design_local in ('GT', 'GC')
        corrected_preds_all = []  # demand forecasts after correction (per item-time)
        true_demand_all = []      # original demand (before substitution) for metric calc

        items_index = pack["items_index"]
        test_times = pack["test_times"]
        rows_sorted_by_t = pack["rows_sorted_by_t"]
        cap_map = pack["cap_map"]
        P_by_store = pack["P_by_store"]
        pred_info_ctrl_all = pack["pred_info_ctrl_all"]
        pred_info_trt_all = pack["pred_info_trt_all"]
        cfg2_local = pack["cfg2"]

        # Pre-fetch WPE once before the experiment loop
        control_wpe_info = None
        if cfg2_local.get('s1_mode', False):
            control_wpe_info = _get_control_wpe(cfg2_local)
        if cfg2_local.get('s2_mode', False):
            control_wpe_info = _get_control_wpe(cfg2_local)
        
        treatment_wpe_info = None
        #  Fix TypeError by loading treatment_wpe for s2_mode ---
        if cfg2_local.get('s2_oracle_mode', False) or cfg2_local.get('s2_mode', False):
            treatment_wpe_info = _get_treatment_wpe(cfg2_local)

        wpe_meta_cols = [col for col in ['store_id','city_id','first_category_id','second_category_id','dayofweek'] if col in work.columns]

        rng = np.random.default_rng(cfg2_local['seed'] + exp_id)

        # assignment
        N = len(items_index); T = len(test_times)
        W = gen_assignment_matrix(N, T, pack["design"], p_model2=float(cfg2_local['p_treat']), seed=cfg2_local['seed'] + exp_id)

        # Detail buffer for this experiment
        save_all_details = bool(cfg2_local.get("save_all_details", False))
        exp_detail_dir = out_dir / "details_by_exp"
        detail_cols = ["store_id","product_id",ITEM_KEY,"time_idx","dt","sale_amount",
                       "exp_id","W","dist_type","mu","sigma","S","I","S_unconstraint","D_true","realized","reward"]
        detail_buf = {k: [] for k in detail_cols}

        for ti, t in enumerate(test_times):
            rows = rows_sorted_by_t[t]
            # Work slice for this test time. Reset index so all per-time arrays share the same positional order.
            df_t = work.loc[rows].copy().reset_index().rename(columns={'index': 'work_row'})

            sid_arr = df_t['store_id'].to_numpy(np.int64)

            item_idx_arr = items_index.merge(  # noqa
                df_t[['store_id', 'product_id']], on=['store_id', 'product_id'],
                how='right', sort=False
            )['item_idx'].to_numpy(np.int64)

            b = df_t['actual_sale_price'].to_numpy(np.float32)
            c = df_t['ordering_cost'].to_numpy(np.float32)
            h = df_t['holding_cost'].to_numpy(np.float32)
            I = df_t['inventory'].to_numpy(np.float32)
            D_true = df_t['sale_amount_pred'].to_numpy(np.float32)

            w_col = W[item_idx_arr, ti].astype(np.int8)

            # Select prediction distributions by assignment
            oracle_like = (cfg2_local.get('oracle_mode', False)
                           or cfg2_local.get('s1_oracle_mode', False)
                           or cfg2_local.get('s2_oracle_mode', False))

            # Align predictions to the current slice by (store_id, product_id, dt) rather than by row position.
            pred_info_t_ctrl = _align_pred_info_to_slice(pred_info_ctrl_all, df_t)
            pred_info_t_trt = None
            if pred_info_trt_all is not None:
                pred_info_t_trt = _align_pred_info_to_slice(pred_info_trt_all, df_t)

            if oracle_like:
                # In oracle modes, control predictions are always needed; treatment predictions are used only by S2 oracle.
                pred_info_t = pred_info_t_ctrl
            else:
                # In standard probabilistic mode, pick per-row prediction by assignment.
                pred_info_t = pred_info_t_ctrl.copy()
                is_treat = (w_col == 1)
                if pred_info_t_trt is not None and np.any(is_treat):
                    pred_cols = [col for col in pred_info_t.columns if col not in ['store_id','product_id','dt']]
                    pred_info_t.loc[is_treat, pred_cols] = pred_info_t_trt.loc[is_treat, pred_cols].to_numpy()
            assert pred_info_t[['store_id','product_id','dt']].equals(df_t[['store_id','product_id','dt']])

            S = np.empty_like(I, dtype=np.float32)
            S_uncon = np.empty_like(I, dtype=np.float32)

            unique_sids = np.unique(sid_arr)
            for sid in unique_sids:
                idx = df_t.index[df_t['store_id'].to_numpy(np.int64) == int(sid)].to_numpy()
                cap = float(cap_map.get(int(sid), math.inf))
                meta_slice = df_t.loc[idx, wpe_meta_cols] if wpe_meta_cols else pd.DataFrame(index=np.arange(len(idx)))
                control_wpe_vec = _resolve_group_wpe_vector(control_wpe_info, meta_slice) if control_wpe_info is not None else np.zeros(len(idx), dtype=float)
                treatment_wpe_vec = _resolve_group_wpe_vector(treatment_wpe_info, meta_slice) if treatment_wpe_info is not None else np.zeros(len(idx), dtype=float)

                #  Implement correct Oracle Mode logic ---
                # In Oracle mode, the Treatment group uses true demand, while the Control group
                # uses its own forecast in deterministic mode.
                if cfg2_local.get('oracle_mode', False):
                    # In Oracle mode, construct a hybrid demand vector `d` for the entire store.
                    d_store = np.zeros_like(I[idx], dtype=float)
                    is_treat_local = (w_col[idx] == 1)
                    is_ctrl_local = ~is_treat_local
                    
                    # 1. For treatment items, use true demand.
                    if np.any(is_treat_local):
                        d_store[is_treat_local] = D_true[idx][is_treat_local]

                    # 2. For control items, use their model's point forecast.
                    if np.any(is_ctrl_local):
                        pred_info_ctrl = pred_info_t.iloc[idx][is_ctrl_local]
                        #  Robustly get point forecast ---
                        # Handle both normal (mu) and quantile (q_0.50) distributions.
                        if 'mu' in pred_info_ctrl.columns:
                            d_store[is_ctrl_local] = pred_info_ctrl['mu'].to_numpy()
                        else:
                            d_store[is_ctrl_local] = pred_info_ctrl['q_0.50'].to_numpy()

                    # 3. Call optimizer once for the whole store with the hybrid demand vector.
                    res = optimize_inventory(
                        pred_info_slice=None, # Not needed as we provide demand_vec
                        I_vec=I[idx], b=b[idx], c=c[idx], h=h[idx], B_cap=cap,
                        demand_vec=d_store,
                        mode='deterministic'
                    )
                    if collect_metrics:
                        corrected_preds_all.append(d_store)
                        true_demand_all.append(D_true[idx])
                elif cfg2_local.get('s1_oracle_mode', False):
                    # S1 Oracle Mode: Treatment is a bias-corrected version of Control.
                    d_store = np.zeros_like(I[idx], dtype=float)
                    is_treat_local = (w_col[idx] == 1)
                    is_ctrl_local = ~is_treat_local

                    # 1. Get the control model's point forecast for all items in the store.
                    pred_info_store = pred_info_t.iloc[idx]
                    if 'mu' in pred_info_store.columns:
                        d_control_store = pred_info_store['mu'].to_numpy()
                    else:
                        d_control_store = pred_info_store['q_0.50'].to_numpy()
                    
                    # 2. Calculate the mean error (bias) for this store at this time step.
                    mean_error = np.mean(d_control_store - D_true[idx])

                    # 3. Assign demand vectors for control and treatment groups.
                    d_store[is_ctrl_local] = d_control_store[is_ctrl_local]
                    d_store[is_treat_local] = d_control_store[is_treat_local] - mean_error
                    d_store = np.maximum(d_store, 0) # Ensure demand is non-negative

                    # 4. Call optimizer once for the whole store with the hybrid demand vector.
                    res = optimize_inventory(
                        pred_info_slice=None, # Not needed as we provide demand_vec
                        I_vec=I[idx], b=b[idx], c=c[idx], h=h[idx], B_cap=cap,
                        demand_vec=d_store, mode='deterministic'
                    )
                    if collect_metrics:
                        corrected_preds_all.append(d_store)
                        true_demand_all.append(D_true[idx])

                elif cfg2_local.get('s1_mode', False):
                    # S1 Mode: Treatment is a WPE-corrected version of Control.
                    d_store = np.zeros_like(I[idx], dtype=float)
                    is_treat_local = (w_col[idx] == 1)
                    is_ctrl_local = ~is_treat_local

                    # 1. Get the control model's point forecast for all items in the store.
                    pred_info_store = pred_info_t.iloc[idx]
                    if 'mu' in pred_info_store.columns:
                        d_control_store = pred_info_store['mu'].to_numpy()
                    else:
                        d_control_store = pred_info_store['q_0.50'].to_numpy()

                    # 2. Assign demand vectors for control and treatment groups.
                    d_store[is_ctrl_local] = d_control_store[is_ctrl_local]
                    # 3. For treatment, apply the pre-calculated ME correction (subtract bias).
                    d_store[is_treat_local] = d_control_store[is_treat_local] - control_wpe_vec[is_treat_local]
                    d_store = np.maximum(d_store, 0) # Ensure demand is non-negative

                    # 4. Call optimizer once for the whole store with the hybrid demand vector.
                    res = optimize_inventory(
                        pred_info_slice=None, # Not needed as we provide demand_vec
                        I_vec=I[idx], b=b[idx], c=c[idx], h=h[idx], B_cap=cap,
                        demand_vec=d_store, mode='deterministic'
                    )
                    if collect_metrics:
                        corrected_preds_all.append(d_store)
                        true_demand_all.append(D_true[idx])
                elif cfg2_local.get('s2_oracle_mode', False):
                    if pred_info_t_trt is None:
                        raise ValueError("s2_oracle_mode requires treatment-model predictions (pred_info_trt_all).")
                    # S2 Oracle Mode: Treatment=D_true, Control=WPE-corrected treatment model
                    d_store = np.zeros_like(I[idx], dtype=float)
                    is_treat_local = (w_col[idx] == 1)
                    is_ctrl_local = ~is_treat_local

                    # 1. For treatment items, use true demand.
                    if np.any(is_treat_local):
                        d_store[is_treat_local] = D_true[idx][is_treat_local]

                    # 2. For control items, use the WPE-corrected forecast from the *treatment* model.
                    if np.any(is_ctrl_local):
                        # We need the treatment model's predictions for the control items
                        pred_info_trt_for_ctrl = pred_info_t_trt.iloc[idx[is_ctrl_local]]
                        if 'mu' in pred_info_trt_for_ctrl.columns:
                            d_treat_forecast = pred_info_trt_for_ctrl['mu'].to_numpy()
                        else:
                            d_treat_forecast = pred_info_trt_for_ctrl['q_0.50'].to_numpy()
                        d_store[is_ctrl_local] = d_treat_forecast - treatment_wpe_vec[is_ctrl_local]

                    d_store = np.maximum(d_store, 0) # Ensure demand is non-negative
                    res = optimize_inventory(
                        pred_info_slice=None, I_vec=I[idx], b=b[idx], c=c[idx], h=h[idx], B_cap=cap,
                        demand_vec=d_store, mode='deterministic'
                    )
                    if collect_metrics:
                        corrected_preds_all.append(d_store)
                        true_demand_all.append(D_true[idx])
                elif cfg2_local.get('s2_mode', False):
                    # S2 Mode: Control and Treatment groups both use their respective models,
                    # corrected by their own pre-calculated WPE.
                    d_store = np.zeros_like(I[idx], dtype=float)
                    is_treat_local = (w_col[idx] == 1)
                    is_ctrl_local = ~is_treat_local

                    # 1. For control items, use the WPE-corrected forecast from the control model.
                    if np.any(is_ctrl_local):
                        pred_info_ctrl_for_ctrl = pred_info_t_ctrl.iloc[idx[is_ctrl_local]]
                        d_ctrl_forecast = pred_info_ctrl_for_ctrl.get('mu', pred_info_ctrl_for_ctrl.get('q_0.50')).to_numpy()
                        d_store[is_ctrl_local] = d_ctrl_forecast - control_wpe_vec[is_ctrl_local]

                    # 2. For treatment items, use the WPE-corrected forecast from the treatment model.
                    if np.any(is_treat_local):
                        pred_info_trt_for_trt = pred_info_t_trt.iloc[idx[is_treat_local]]
                        d_treat_forecast = pred_info_trt_for_trt.get('mu', pred_info_trt_for_trt.get('q_0.50')).to_numpy()
                        d_store[is_treat_local] = d_treat_forecast - treatment_wpe_vec[is_treat_local]

                    d_store = np.maximum(d_store, 0) # Ensure demand is non-negative
                    res = optimize_inventory(
                        pred_info_slice=None, I_vec=I[idx], b=b[idx], c=c[idx], h=h[idx], B_cap=cap,
                        demand_vec=d_store, mode='deterministic'
                    )
                    if collect_metrics:
                        corrected_preds_all.append(d_store)
                        true_demand_all.append(D_true[idx])

                else:
                    pred_info_store = pred_info_t.iloc[idx]
                    if 'mu' in pred_info_store.columns:
                        d_store = pred_info_store['mu'].to_numpy()
                    else:
                        d_store = pred_info_store['q_0.50'].to_numpy()

                    res = optimize_inventory(
                        pred_info_slice=None,
                        I_vec=I[idx], b=b[idx], c=c[idx], h=h[idx], B_cap=cap,
                        demand_vec=d_store, mode='deterministic'
                    )
                    if collect_metrics:
                        corrected_preds_all.append(d_store)
                        true_demand_all.append(D_true[idx])

                S_uncon[idx] = res['S_uncon'].astype(np.float32, copy=False)
                S[idx]       = res['S_con'].astype(np.float32, copy=False)

            order_qty = np.maximum(S - I, 0.0)
            overshoot_flags.extend((order_qty <= 1e-9).tolist())

            incoming_all = np.zeros_like(D_true, dtype=float)
            if int(cfg2_local['stockout_substitute']) == 0:
                realized = np.minimum(S, D_true)
            else:
                realized = np.minimum(S, D_true).astype(np.float32, copy=False)
                for sid in unique_sids:
                    idx_local = np.where(sid_arr == sid)[0]
                    if idx_local.size <= 1:
                        continue
                    Dg = D_true[idx_local].copy(); Sg = S[idx_local].copy()
                    short = np.maximum(Dg - Sg, 0.0)
                    incoming = np.zeros_like(short)
                    P = P_by_store.get(int(sid))
                    if P is not None:
                        for i, over in enumerate(np.round(short).astype(int)):
                            if over <= 0:
                                continue
                            probs = P[i].copy()
                            ssum = probs.sum()
                            if ssum > 0:
                                probs = probs / ssum
                                incoming += rng.multinomial(over, probs)
                    D_aug = Dg + incoming
                    incoming_all[idx_local] = incoming
                    D_true[idx_local] = D_aug
                    realized[idx_local] = np.minimum(Sg, D_aug)
            final_demand = D_true
            mask = final_demand > 0
            if np.any(mask):
                transfer_ratios.extend((incoming_all[mask] / final_demand[mask]).tolist())

            rem = np.maximum(S - realized, 0.0).astype(np.float32, copy=False)
            reward = (b * realized - c * np.maximum(S - I, 0.0) - h * rem).astype(np.float32)
            if ti == T - 1:
                reward = (reward + c * rem).astype(np.float32)

            # Carry remaining inventory to next period
            if ti < T - 1:
                next_t = test_times[ti + 1]
                sids = df_t['store_id'].astype(int).tolist()
                pids = df_t['product_id'].astype(int).tolist()
                next_keys = [f"{s}_{p}_{next_t}" for s, p in zip(sids, pids)]
                mapping_next = key_to_rows_by_t[next_t]
                next_rows = np.fromiter((mapping_next.get(k, -1) for k in next_keys), dtype=int, count=len(next_keys))
                valid = next_rows >= 0
                if np.any(valid):
                    work.loc[next_rows[valid], 'inventory'] = rem[valid]

            detail_buf["store_id"].extend(df_t["store_id"].astype(int).tolist())
            detail_buf["product_id"].extend(df_t["product_id"].astype(int).tolist())
            detail_buf["item_key"].extend(df_t[ITEM_KEY].tolist())
            detail_buf["time_idx"].extend(df_t["time_idx"].astype(int).tolist())
            detail_buf["dt"].extend(df_t["dt"].tolist())
            detail_buf["sale_amount"].extend(D_true.tolist())
            detail_buf["exp_id"].extend([exp_id]*len(rows))
            buf_w = w_col.tolist()
            detail_buf["W"].extend(buf_w)
            detail_buf["dist_type"].extend(pred_info_t['dist_type'].tolist())
            detail_buf["mu"].extend(pred_info_t.get('mu', pd.Series(np.nan, index=pred_info_t.index)).tolist())
            detail_buf["sigma"].extend(pred_info_t.get('sigma', pd.Series(np.nan, index=pred_info_t.index)).tolist())
            detail_buf["S"].extend(S.tolist())
            detail_buf["I"].extend(I.tolist())
            detail_buf["S_unconstraint"].extend(S_uncon.tolist())
            detail_buf["D_true"].extend(D_true.tolist())
            detail_buf["realized"].extend(realized.tolist())
            detail_buf["reward"].extend(reward.tolist())


        det_exp_df = pd.DataFrame(detail_buf)
        det_exp_df.sort_values([ITEM_KEY,'dt'], inplace=True, ignore_index=True)
        det_exp_path = exp_detail_dir / f"detail_exp_{exp_id}.parquet"  # return path only
        if exp_id == 0 or save_all_details:
            det_exp_df.to_parquet(exp_detail_dir / f"detail_exp_{exp_id}.parquet", index=False)

        treat_mean = det_exp_df.loc[det_exp_df['W'] == 1, 'reward'].mean()
        ctrl_mean  = det_exp_df.loc[det_exp_df['W'] == 0, 'reward'].mean()
        gte_hat = float(treat_mean - ctrl_mean) if (pd.notna(treat_mean) and pd.notna(ctrl_mean)) else None
        transfer_ratio_mean = float(np.mean(transfer_ratios)) if transfer_ratios else None
        overshoot_ratio_mean = float(np.mean(overshoot_flags)) if overshoot_flags else None
        wpe_metrics = {"WAPE": None, "WPE": None, "ME": None}
        if collect_metrics and corrected_preds_all and true_demand_all:
            y_pred_corr = np.concatenate(corrected_preds_all).astype(float, copy=False)
            y_true_base = np.concatenate(true_demand_all).astype(float, copy=False)
            wpe_metrics = _calc_wape_wpe_me(y_true_base, y_pred_corr)

        return {
            "exp_id": exp_id,
            "treat_mean": float(treat_mean) if pd.notna(treat_mean) else None,
            "control_mean": float(ctrl_mean) if pd.notna(ctrl_mean) else None,
            "gte_hat": gte_hat,
            "transfer_ratio_mean": transfer_ratio_mean,
            "overshoot_ratio_mean": overshoot_ratio_mean,
            "wape_corrected": wpe_metrics["WAPE"],
            "wpe_corrected": wpe_metrics["WPE"],
            "me_corrected": wpe_metrics["ME"],
            "detail_df": det_exp_df if (exp_id == 0 or save_all_details) else None
        }

    # results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
    #     delayed(_run_single_experiment)(exp_id, readonly_pack) for exp_id in range(num)
    # )
    results = []
    for exp_id in range(num):
        results.append(_run_single_experiment(exp_id, readonly_pack))

    # ---------- 4) Summaries and persistence ----------
    # Save experiment summary
    exp_summary = pd.DataFrame(results)
    exp_summary.insert(0, 'design', design)
    # Across-experiment average of transfer ratio
    transfer_mean_all = exp_summary['transfer_ratio_mean'].dropna().mean() if 'transfer_ratio_mean' in exp_summary.columns else float("nan")
    exp_summary['transfer_ratio_mean_all_exps'] = transfer_mean_all if not np.isnan(transfer_mean_all) else None
    overshoot_mean_all = exp_summary['overshoot_ratio_mean'].dropna().mean() if 'overshoot_ratio_mean' in exp_summary.columns else float("nan")
    exp_summary['overshoot_ratio_mean_all_exps'] = overshoot_mean_all if not np.isnan(overshoot_mean_all) else None
    # Propagate corrected metrics (computed only when collect_metrics=True) to all rows
    for col in ['wape_corrected', 'wpe_corrected', 'me_corrected']:
        if col in exp_summary.columns:
            col_mean = exp_summary[col].dropna().mean()
            exp_summary[col] = col_mean if not np.isnan(col_mean) else None
    exp_summary.to_csv(out_dir / "exp_summary.csv", index=False)
    
    all_logs = None
    detail_dfs = [r['detail_df'] for r in results if r['detail_df'] is not None]
    if detail_dfs:
        all_logs = pd.concat(detail_dfs, axis=0, ignore_index=True)
        all_logs.sort_values([ITEM_KEY,'dt'], inplace=True, ignore_index=True)
        all_logs.to_parquet(out_dir / "detail_logs.parquet", index=False)



    # Save cfg2
    cfg2_serializable = cfg2.copy()
    if 'path' in cfg2_serializable and isinstance(cfg2_serializable['path'], Path):
        cfg2_serializable['path'] = str(cfg2_serializable['path'])
    with open(out_dir / "ab_config2.json", "w", encoding="utf-8") as f:
        json.dump(cfg2_serializable, f, ensure_ascii=False, indent=2)

    print(f"[OK] {design}: exp_summary -> {out_dir/'exp_summary.csv'} ; details in {out_dir / 'details_by_exp'}")
    return OneDesignResult(exp_summary, all_logs, out_dir)

# def run_all_designs(cfg2: dict, capacity_df: pd.DataFrame):
def run_all_designs(cfg2: dict, capacity_df: pd.DataFrame, price_enhanced_eval_df: pd.DataFrame):
    """
    Steps:
      1) Run GT / GC to compute gte_true = mean(GT.treat_mean) - mean(GC.control_mean)
      2) Run SW / IR / PR (each design runs run_one_design in parallel over exp_id)
      3) Aggregate gte_hat for each design and write abtest_summary.csv (mean/var/std/bias/mse)
      4) Plot violin chart to figs/gte_violin.png
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # ---- 1) GT & GC: force each to run once to get gte_true ----
    print("[RUN] Running GT to compute gte_true")
    gt_res = run_one_design('GT', cfg2, capacity_df, price_enhanced_eval_df)
    print("[RUN] Running GC to compute gte_true")
    gc_res = run_one_design('GC', cfg2, capacity_df, price_enhanced_eval_df)

    # GT has only treatment; GC has only control — take their means separately
    gt_treat_mean = gt_res.exp_summary['treat_mean'].dropna().mean()
    gc_ctrl_mean  = gc_res.exp_summary['control_mean'].dropna().mean()
    #  Print intermediate GT/GC results ---
    print(f"  -> GT treat_mean: {gt_treat_mean:.6g}")
    print(f"  -> GC control_mean: {gc_ctrl_mean:.6g}")
    # -----------
    # ------------------------------------------------
    gte_true = float(gt_treat_mean - gc_ctrl_mean)

    # Save gte_true
    figs_dir = cfg2['path'] / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg2['path'] / "gte_true.txt", "w", encoding="utf-8") as f:
        f.write(f"{gte_true:.6g}\n")
    print(f"[OK] GTE_true = {gte_true:.6g}")

    # ---- 2) Run SW / IR / PR (or defaults if not specified) ----
    wanted = [d for d in cfg2.get('designs', ['SW','IR','PR']) if d in ('SW','IR','PR')]
    # wanted = [d for d in cfg2.get('designs', ['IR','PR']) if d in ('IR','PR')]
    if not wanted:
        wanted = ['SW','IR','PR'] 

    design_results = {}
    for d in wanted:
        print(f"[RUN] Running design: {d}")
        design_results[d] = run_one_design(d, cfg2, capacity_df, price_enhanced_eval_df)

    # ---- 3) Aggregate to abtest_summary.csv (mean/var/std/bias/MSE) ----
    rows = [{
        "kind": "GTE_true",
        "design": "GT-GC",
        "n": 1,
        "p_treat": 1.0,
        "gte_true": gte_true,
        "treat_mean": gt_treat_mean,
        "control_mean": gc_ctrl_mean,
        "gte_hat_mean": gte_true,
        "gte_hat_var": float("nan"),
        "gte_hat_std": float("nan"),
        "bias": 0.0,
        "mse": 0.0
    }]

    violin_vals, violin_labels = [], []
    for d in ['SW', 'IR', 'PR']:
        if d not in design_results:
            continue
        res = design_results[d]
        vals = res.exp_summary['gte_hat'].dropna().to_numpy(dtype=float)
        n = int(vals.size)
        if n > 0:
            mean = float(np.mean(vals))
            var  = float(np.var(vals, ddof=1)) if n > 1 else float("nan")
            std  = float(np.sqrt(var)) if n > 1 else float("nan")
            bias = float(mean - gte_true)
            mse  = float((mean - gte_true)**2 + (var if np.isfinite(var) else 0.0))
            violin_vals.append(vals)
            violin_labels.append(d)
        else:
            mean = var = std = bias = mse = float("nan")

        rows.append({
            "kind": "randomized_design",
            "design": d,
            "n": n,
            "p_treat": float(cfg2['p_treat']),
            "gte_true": gte_true,
            "treat_mean": float("nan"), # For randomized designs, these are not directly comparable to GT/GC
            "control_mean": float("nan"),
            "gte_hat_mean": mean,
            "gte_hat_var": var,
            "gte_hat_std": std,
            "bias": bias,
            "mse": mse
        })

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(cfg2['path'] / f"abtest_summary_{cfg2['model_shorthand']}.csv", index=False)
    print(f"[OK] Summary saved at {cfg2['path']}/'abtest_summary_{cfg2['model_shorthand']}.csv'")

    # ---- 4) Plot violin: SW/IR/PR gte_hat distribution + gte_true line ----
    if len(violin_vals) > 0:
        fig, ax = plt.subplots(figsize=(5, 4.2), dpi=140)
        parts = ax.violinplot(violin_vals, showmeans=True, showextrema=True, showmedians=False)
        ax.axhline(gte_true, linestyle="--", linewidth=1.5, color="tab:gray")
        ax.set_xticks(np.arange(1, len(violin_labels) + 1))
        ax.set_xticklabels(violin_labels, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        # ax.set_ylim([0, 12])
        fig.tight_layout()
        fig.savefig(figs_dir / f"gte_violin_{cfg2['model_shorthand']}.png")
        plt.close(fig)
        print(f"[OK] Violin plot saved at {figs_dir}/'gte_violin_{cfg2['model_shorthand']}.png'")
    else:
        print("[WARN] No gte_hat values found to plot violin.")

    return {
        "gte_true": gte_true,
        "summary": df_sum,
        "design_outputs": {k: str(v.out_dir) for k, v in design_results.items()},
        "GT_out": gt_res.out_dir,
        "GC_out": gc_res.out_dir
    }


# endregion

# region inventory price capacity
from typing import Dict
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def optimize_inventory( # noqa
    pred_info_slice: pd.DataFrame,
    I_vec: np.ndarray,
    b: np.ndarray, c: np.ndarray, h: np.ndarray, B_cap: float, demand_vec: Optional[np.ndarray] = None,
    mode: str = 'deterministic', max_iter: int = 100, tol: float = 1e-4
) -> Dict[str, np.ndarray]:
    """ 
    Newsvendor with store capacity (via Lagrange multiplier bisection).
    - pred_info_slice: distribution info for all items at a time step.
    - mode: 'probabilistic' (default) for newsvendor, or 'deterministic' for point forecast optimization.

    - demand_vec: optional deterministic demand vector when mode='deterministic'.
        - normal: requires 'mu' and 'sigma' columns.
        - quantiles: requires 'q_0.05', 'q_0.10', ... columns.
        - must include 'dist_type'.
    Returns:
      {
        'S_uncon': unconstrained optimum per item (clipped by I),
        'S_con'  : optimum under capacity B_cap
      }
    """
    I = np.asarray(I_vec, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    h = np.asarray(h, dtype=float)

    if mode == 'deterministic':
        # --- Deterministic Demand Optimization ---
        if demand_vec is not None:
            d = demand_vec
        else:
            d = np.zeros_like(I, dtype=float)
            normal_mask = (pred_info_slice['dist_type'] == 'normal').to_numpy()
            tft_mask = (pred_info_slice['dist_type'] == 'quantiles').to_numpy()
            if np.any(normal_mask):
                d[normal_mask] = pred_info_slice.loc[normal_mask, 'mu'].to_numpy()
            if np.any(tft_mask):
                d[tft_mask] = pred_info_slice.loc[tft_mask, 'q_0.50'].to_numpy()

        # Unconstrained solution is simply the demand
        S_uncon = np.maximum(d, I)

        if np.sum(S_uncon) <= B_cap:
            return {'S_uncon': S_uncon, 'S_con': S_uncon}

        # Constrained solution using a greedy approach
        S_con = I.copy()
        B_rem = B_cap - np.sum(S_con)

        if B_rem > 0:
            # Phase 1: Fill up to demand d_n, sorted by profit margin b_n - c_n
            profit_margin = b - c
            delta_to_demand = np.maximum(0, d - S_con)
            
            # Sort items by descending profit margin
            order = np.argsort(-profit_margin)
            
            for i in order:
                if B_rem <= 0: break
                alloc = min(delta_to_demand[i], B_rem)
                S_con[i] += alloc
                B_rem -= alloc

        # if B_rem > 0:
        #     # Phase 2: If capacity remains, overstock the item with the lowest holding cost
        #     # Sort items by ascending holding cost
        #     order = np.argsort(h)
        #     best_item_to_overstock = order[0]
        #     S_con[best_item_to_overstock] += B_rem
 
        return {'S_uncon': S_uncon, 'S_con': S_con}

    # --- Probabilistic Demand Optimization (Original Logic) ---
    if mode != 'probabilistic':
        raise ValueError("Invalid mode. Choose 'probabilistic' or 'deterministic'.")

    # Effective underage cost b-c
    under = np.clip(b - c, 1e-8, None)
    q0 = np.clip(under / (under + h), 1e-6, 1 - 1e-6)

    # Precompute TFT quantiles once per step
    tft_mask = (pred_info_slice['dist_type'] == 'quantiles').to_numpy()
    if np.any(tft_mask):
        quantile_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        quantile_cols = [f"q_{q:.2f}" for q in quantile_levels]
        tft_quantile_values = pred_info_slice.loc[tft_mask, quantile_cols].to_numpy(dtype=float)

    def calculate_S_vectorized(q_vec: np.ndarray, pred_info: pd.DataFrame) -> np.ndarray:
        S = np.zeros_like(q_vec, dtype=float)
        
        normal_mask = (pred_info['dist_type'] == 'normal').to_numpy()
        if np.any(normal_mask):
            mu = pred_info.loc[normal_mask, 'mu'].to_numpy()
            sigma = pred_info.loc[normal_mask, 'sigma'].to_numpy() 
            S[normal_mask] = norm.ppf(q_vec[normal_mask], loc=mu, scale=np.maximum(sigma, 1e-6))

        if np.any(tft_mask):
            S[tft_mask] = np.array([np.interp(q, quantile_levels, values) for q, values in zip(q_vec[tft_mask], tft_quantile_values)])

        return S

    S_uncon_raw = calculate_S_vectorized(q0, pred_info_slice)
    S_uncon = np.maximum(S_uncon_raw, I).astype(np.float64)

    def compute_S(lam: float) -> np.ndarray:
        q = np.clip((under - lam) / (under + h), 1e-6, 1 - 1e-6)
        S_raw = calculate_S_vectorized(q, pred_info_slice)
        return np.maximum(S_raw, I).astype(np.float64)

    S_loose = compute_S(0.0)
    if float(np.sum(S_loose)) <= B_cap:
        return {'S_uncon': S_uncon, 'S_con': S_loose}

    def g(lam: float) -> float:
        return np.sum(compute_S(lam)) - B_cap

    try:
        high_bound = float(np.max(under)) - 1e-6
        if high_bound <= 0:
            lam_sol = 0.0
        else:
            lam_sol = brentq(g, 0.0, high_bound, xtol=tol, maxiter=max_iter)
        S_hat = compute_S(lam_sol)
    except (ValueError, RuntimeError):
        S_hat = S_loose.copy()

    return {'S_uncon': S_uncon, 'S_con': S_hat}

def make_capacity(df_in: pd.DataFrame, path: Path, multiplier: float = 1.5):
    """
    Reads capacity data from a parquet file if it exists, otherwise calculates and saves it.

    Args:
        df_in (pd.DataFrame): Input DataFrame (e.g., test_df) to calculate capacity from if file not found.
        path (Path): The directory path where the capacity file is expected or will be saved.
        multiplier (float): Multiplier used in capacity calculation if the file is not found.
    """
    file_path = path / f"capacity_{multiplier}.parquet"

    try:
        # Attempt to read the capacity file
        cap = pd.read_parquet(file_path)
        print(f"✅ Loaded capacity parquet from: {file_path.name}")
        return cap

    except FileNotFoundError:
        print(f"⚠️ Local capacity file not found: {file_path.name}\nComputing and saving capacity data...")
        path.mkdir(parents=True, exist_ok=True)

        per_item_mean = df_in.groupby([ITEM_KEY])["sale_amount"].mean()
        base = float(per_item_mean.median()) * multiplier
        items_per_store = df_in.groupby("store_id")["product_id"].nunique()
        cap = (items_per_store * base).rename("capacity").reset_index()
        cap["capacity"] = cap["capacity"].astype(float)

        cap.to_parquet(file_path, index=False)
        print(f"✅ Capacity file computed and saved to: {file_path.name}")
        return cap
    except Exception as e:
        print(f"An error occurred while reading or creating capacity data: {e}")
        raise # Re-raise the exception if it's not FileNotFoundError

def generate_price_and_costs(config: Dict, df_in: pd.DataFrame) -> pd.DataFrame:
    """Generate price and cost columns from the provided DataFrame."""
    np.random.seed(config['seed'])
    df = df_in.copy()

    if not pd.api.types.is_datetime64_any_dtype(df['dt']):
        df['dt'] = pd.to_datetime(df['dt'])

    if 'discount' not in df.columns:
        df['discount'] = 1.0

    def generate_random_factor(unique_keys, loc=1.0, scale=0.05):
        factors = np.random.uniform(loc - 2*scale, loc + 2*scale, size=len(unique_keys))
        return dict(zip(unique_keys, factors))

    product_base_map = generate_random_factor(df['product_id'].unique(), loc=50, scale=20)
    df['base_price'] = df['product_id'].map(product_base_map).clip(lower=1)

    if 'first_category_id' in df.columns:
        category_map = generate_random_factor(df['first_category_id'].unique(), loc=1.0, scale=0.1)
        df['category_adj'] = df['first_category_id'].map(category_map)
    else:
        df['category_adj'] = 1.0

    store_map = generate_random_factor(df['store_id'].unique(), loc=1.0, scale=0.05)
    df['store_adj'] = df['store_id'].map(store_map)

    df['price_temp'] = df['base_price'] * df['category_adj'] * df['store_adj']

    if 'holiday_flag' in df.columns:
        df['holiday_adj'] = np.where(df['holiday_flag'] == 1, np.random.uniform(0.98, 1.02, size=len(df)), 1.0)
    else:
        df['holiday_adj'] = 1.0

    stock_col = 'stock_hour6_22_cnt'
    # if stock_col in df.columns:
    #     stock_max = max(16, float(df[stock_col].max()) or 16)
    #     stock_min_adj, stock_max_adj = 1.10, 0.95
    #     df['stock_adj'] = stock_min_adj - (stock_min_adj - stock_max_adj) * (df[stock_col] / stock_max)
    # else:
    #     df['stock_adj'] = 1.0
    df['stock_adj'] = 1.0

    df['price'] = (df['price_temp'] * df['holiday_adj'] * df['stock_adj']).round(2)
    df['ordering_cost'] = (df['price'] * np.random.uniform(0.30, 0.60, size=len(df))).round(2)
    df['holding_cost'] = (df['ordering_cost'] * np.random.uniform(0.00, 0.30, size=len(df))).round(2)
    df['actual_sale_price'] = (df['price'] * df['discount']).round(2)

    df = df.drop(columns=['base_price', 'category_adj', 'store_adj', 'price_temp', 'holiday_adj', 'stock_adj'], errors='ignore')
    return df

def make_price_data(config: dict, path: Path) -> pd.DataFrame:
    """
    If the price enhancement file is missing, generate and save it; otherwise load it.
    """
    file_path = path / f"eval_price_df_{config['seed']}.parquet"
    try:
        df_price = pd.read_parquet(file_path)
        print(f"✅ Loaded price/cost data from: {file_path.name}")
        return df_price
    except FileNotFoundError:
        print(f"⚠️ Price/cost file not found: {file_path.name}\nGenerating and saving...")
        path.mkdir(parents=True, exist_ok=True)

        #  Load dataset from local disk, consistent with other modules ---
        # The path is relative to the script's location in the 'abtest' directory.
        dataset = load_from_disk(PROJECT_ROOT / "frn_50k_local_dataset")
        eval_df = dataset['eval'].to_pandas()
        # ------------------------------------------------------------------------------------

        df_price = generate_price_and_costs(config, eval_df)
        df_price.to_parquet(file_path, index=False)
        print(f"✅ Price/cost data generated and saved to: {file_path.name}")
        return df_price
    except Exception as e:
        print(f"Error processing price/cost data: {e}")
        raise

# endregion

# region main

# ---------- Run ----------
if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Run A/B test simulations.")
    parser.add_argument('--recovery_control', type=str, default='raw', help='Recovery model for the control group.')
    parser.add_argument('--forecast_control', type=str, default='SSA', help='Forecast model for the control group.')
    parser.add_argument('--recovery_treat', type=str, default='ImputeFormer', help='Recovery model for the treatment group.')
    parser.add_argument('--forecast_treat', type=str, default='SSA', help='Forecast model for the treatment group.')
    parser.add_argument('--stockout_substitute', type=int, default=1, choices=[0, 1], help='Enable (1) or disable (0) stockout substitution.')
    parser.add_argument('--capacity_type', type=str, default='medium', choices=['loose', 'medium', 'tight'], help='Capacity level for the simulation.')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs for joblib. Default is 8. Use -1 for all cores.')
    parser.add_argument('--true_demand_source', type=str, default='ImputeFormer', help='Source for the ground truth demand (D_true). Options: raw, DLinear, TimesNet, etc.')
    parser.add_argument('--oracle_mode', action='store_true', help='Enable oracle mode for inventory optimization, using D_true directly.')
    parser.add_argument('--s1_oracle_mode', action='store_true', help='Enable S1 oracle mode (bias-corrected control model for treatment).')
    parser.add_argument('--s1_mode', action='store_true', help='Enable S1 mode (WPE-corrected control model for treatment).')
    parser.add_argument('--s2_mode', action='store_true', help='Enable S2 mode (Both groups use their own WPE-corrected models).')
    parser.add_argument('--s2_oracle_mode', action='store_true', help='Enable S2 oracle mode (Treatment=D_true, Control=WPE-corrected treatment model).')
    args = parser.parse_args()

    # --- Experiment config ---
    ab_test_seed = 251221   # 251103 251025 251202 251204 251221
    np.random.seed(ab_test_seed)

    # Define Treatment and Control groups (from CLI args)
    # recovery_model: 'raw', 'DLinear', 'TimesNet', 'ImputeFormer'
    # forecast_model: 'SSA', 'TFT', 'DLinear'
    control_group = {
        'recovery': args.recovery_control,
        'forecast': args.forecast_control
    }
    treat_group = {
        'recovery': args.recovery_treat,
        'forecast': args.forecast_treat
    }

    # Define capacity and experiment design from CLI args
    capacity_type = args.capacity_type
    if args.s2_mode or args.s2_oracle_mode:
        multiplier_map = {'loose': 1.8, 'medium': 0.9, 'tight': 0.6}
    elif args.s1_mode or args.s1_oracle_mode:
        multiplier_map = {'loose': 1.8, 'medium': 1.2, 'tight': 0.9}
    else:
        # default fallback (choose one)
        multiplier_map = {'loose': 1.8, 'medium': 1.2, 'tight': 0.9}
    capacity_multiplier = multiplier_map[capacity_type]


    # --- Build output directory hierarchy ---
    # 1. Create parameter shorthand string
    model_shorthand = (
        f"{'ORACLE_' if args.oracle_mode else ''}"
        f"{'S1_ORACLE_' if args.s1_oracle_mode else ''}"
        f"{'S1_' if args.s1_mode else ''}"
        f"{'S2_ORACLE_' if args.s2_oracle_mode else ''}"
        f"{'S2_' if args.s2_mode else ''}"
        f"Dtrue_{args.true_demand_source}_"
        f"C_{args.recovery_control}_{args.forecast_control}_"
        f"T_{args.recovery_treat}_{args.forecast_treat}_"
        f"sub_{args.stockout_substitute}_"
        f"cap_{capacity_multiplier}"
    )

    # 2. Construct shared and capacity-specific paths
    base_output_path = PROJECT_ROOT / 'abtest' / 'abtest_outputs' / str(ab_test_seed) / model_shorthand
    path_abtest_save = PROJECT_ROOT / 'abtest' / 'abtest_outputs' / str(ab_test_seed) / model_shorthand / f'{capacity_type}_{capacity_multiplier}'
    path_abtest_save.mkdir(parents=True, exist_ok=True)

    config2 = {
        "recovery_control": control_group['recovery'],
        "forecast_control": control_group['forecast'],
        "recovery_treat": treat_group['recovery'],
        "forecast_treat": treat_group['forecast'],
        "p_treat": 0.5,
        "designs": ["GT", "GC", "SW", "IR", "PR"],
        "stockout_substitute": args.stockout_substitute,
        "num_experiments": 30,
        "capacity": capacity_type,
        'multiplier': capacity_multiplier,
        "seed": ab_test_seed,
        "path": path_abtest_save,
        "n_jobs": args.n_jobs,
        "true_demand_source": args.true_demand_source,
        "oracle_mode": args.oracle_mode,
        "s1_oracle_mode": args.s1_oracle_mode,
        "s1_mode": args.s1_mode,
        "s2_mode": args.s2_mode,
        "s2_oracle_mode": args.s2_oracle_mode,
        "model_shorthand": model_shorthand,
    }

    # Save experiment config
    config2_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in config2.items()}
    with open(path_abtest_save / "ab_config.json", "w", encoding="utf-8") as f:
        json.dump(config2_serializable, f, indent=4)

    # --- Prepare data and run ---
    print("--- Preparing Data for A/B Test ---")
    if args.oracle_mode:
        print("🔮 Oracle mode is ENABLED. Inventory optimization will use true demand (D_true) directly.")
    elif args.s1_oracle_mode:
        print("💡 S1 Oracle (bias-corrected) mode is ENABLED.")
    elif args.s1_mode:
        print("📈 S1 (WPE-corrected) mode is ENABLED.")
    elif args.s2_mode:
        print("📈 S2 (Dual WPE-corrected) mode is ENABLED.")
    elif args.s2_oracle_mode:
        print("💡 S2 Oracle mode is ENABLED. Treatment=D_true, Control=WPE-corrected treatment model.")
    else:
        print("Forecasting mode is ENABLED. Inventory optimization will use model predictions.")
    price_enhanced_eval_df = make_price_data(config2, base_output_path)
    price_enhanced_eval_df[ITEM_KEY] = price_enhanced_eval_df['store_id'].astype(str) + '_' + price_enhanced_eval_df['product_id'].astype(str)
    
    capacity_df = make_capacity(price_enhanced_eval_df, base_output_path, multiplier=config2['multiplier'])

    print("\n--- Starting A/B Test Simulation ---")
    results = run_all_designs(config2, capacity_df, price_enhanced_eval_df)
    print("\n--- A/B Test Simulation Finished ---")


# endregion