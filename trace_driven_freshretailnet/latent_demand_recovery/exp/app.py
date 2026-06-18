import os, subprocess, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.append(project_root)

import torch
from torch import nn
import pypots
from data import load_data
from model import load_model
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)
from datetime import datetime
from datasets import load_from_disk
import argparse

def set_seed(seed_value=1024):
    np.random.seed(seed_value)
    random.seed(seed_value)
    # 设置CPU的种子
    torch.manual_seed(seed_value)
    # 如果你使用的是CUDA，还需要设置CUDA的种子
    torch.cuda.manual_seed(seed_value)
    # 如果你使用的是多GPU，还需要设置随机种子的 all-gather 方法
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    return


def _imputation(CONFIG):
    print("\n[RECOVERY] Step 1: Starting imputation process...")
    # load data
    data = load_data(CONFIG)
    (
        train_set, 
        ts_origin, 
        valid_idx,
    ) = (
        data['train_set'],
        data['ts_origin'], 
        data['valid_idx'],
    )
    # update specific dataset params config
    CONFIG.update(data['params'])
    model = load_model(CONFIG)
    print(f"[RECOVERY] Fitting model '{CONFIG['model']}' on the training data...")
    model.fit(train_set)
    print("[RECOVERY] Predicting on the training data (for MNAR evaluation or initial recovery)...")
    results = model.predict(train_set)
    if len(results['imputation'].shape) == 4: # Handle models that return multiple imputations
        imputation = results['imputation'].mean(axis=1)[:, :, :CONFIG['OT']]
    else:
        imputation = results['imputation'][:,:,:CONFIG['OT']]
    imputation = np.where(imputation>0, imputation, 0)
    model_name = CONFIG['model']
    missing_rate = CONFIG['missing_rate']
    if not os.path.exists('./demand'):
        os.makedirs('./demand', exist_ok=True)
    
    imputation_filename = f'./demand/{model_name}_imputation_{missing_rate}.npy'
    np.save(imputation_filename, imputation)
    print(f"[RECOVERY] Saved initial imputation numpy array to: {imputation_filename}")

    if CONFIG['missing_rate']>0:
        evaluation_mnar(train_set['X'], imputation, CONFIG)
    return imputation, model
    
def _demand_recovery(imputation, CONFIG):
    print("\n[RECOVERY] Step 2: Reconstructing full demand for the TRAIN set...")
    # --- Gemini Code Assist: Load dataset from local disk ---
    dataset = load_from_disk("../../frn_50k_local_dataset")
    # ------------------------------------------------------
    data = dataset['train'].to_pandas()
    data = data.sort_values(by=['store_id', 'product_id', 'dt'])
    horizon=90
    series_num = data.shape[0]//horizon
    hours_sale = np.array(data['hours_sale'].tolist())
    hours_sale_origin = hours_sale.reshape(series_num*3, 30, 24)
    hours_sale_origin[...,6:22] = imputation.reshape(-1, 30, 16)
    sale_amount_pred = hours_sale_origin.sum(axis=-1).reshape(-1, 90)
    data[f'sale_amount_pred'] = sale_amount_pred.reshape(-1)
    if not os.path.exists('./demand'):
        os.makedirs('./demand', exist_ok=True)
    # --- Gemini Code Assist: Save parquet with model name to avoid overwriting ---
    model_name = CONFIG['model']
    output_filename = f'./demand/demand_{model_name}.parquet'
    data.to_parquet(output_filename)
    print(f"[RECOVERY] Saved recovered TRAIN set demand to: {output_filename}")
    return data
    
def demand_recovery(CONFIG):
    imputation, model = _imputation(CONFIG)

    # --- Gemini Code Assist: Always perform full demand recovery regardless of missing_rate ---
    # The MNAR evaluation (if missing_rate > 0) is a separate analysis.
    # The main goal is to get the recovered demand for both train and eval sets.
    
    # Recover demand for the training set
    demand_df = _demand_recovery(imputation, CONFIG)
    evaluation_decoupling(demand_df, CONFIG, data_split='train')
    evaluation_non_stockout_consistency(demand_df, CONFIG, data_split='train')
    
    # Recover demand for the evaluation set
    impute_eval_and_save(CONFIG, model)
    return demand_df


def impute_eval_and_save(CONFIG, model):
    """Uses the trained imputer to recover demand on the eval set and saves the results."""
    print("\n[RECOVERY] Step 3: Reconstructing full demand for the EVAL set...")
    # --- Gemini Code Assist: Adopted user's robust logic for eval set imputation ---
    # This implementation correctly constructs a sliding window for prediction,
    # matching the data structure the pypots models were trained on.

    print("[RECOVERY] Loading train and eval splits for evaluation recovery...")
    dataset = load_from_disk("../../frn_50k_local_dataset")

    # Load and sort data to ensure consistent series order
    df_tr = dataset['train'].to_pandas().sort_values(['store_id', 'product_id', 'dt'])
    df_ev = dataset['eval'].to_pandas().sort_values(['store_id', 'product_id', 'dt'])

    # Get series dimensions
    series_num = df_tr.shape[0] // 90
    assert df_ev.shape[0] // 7 == series_num, "Train/eval series count mismatch."

    # Helper to prepare hourly data blocks
    def prepare_hourly_data(df, days):
        """Prepares hourly sales, stock status, and covariates."""
        hrs = np.array(df['hours_sale'].tolist()).reshape(series_num, days, 24)
        st = np.array(df['hours_stock_status'].tolist()).reshape(series_num, days, 24)
        cov4 = df[['discount', 'holiday_flag', 'precpt', 'avg_temperature']].values.reshape(series_num, days, 4)
        hrs16 = hrs[:, :, 6:22]
        st16 = st[:, :, 6:22]
        # Mask sales with NaN where stockouts occurred
        x = np.where(st16 == 1, np.nan, hrs16) # This is the 'impute' mode logic
        return x, cov4, hrs

    # Prepare data blocks using the default 'impute' logic
    x_tr, c_tr, _ = prepare_hourly_data(df_tr, 90)
    x_ev, c_ev, hrs_sale_eval_origin_24h = prepare_hourly_data(df_ev, 7)

    # Take the last 30 days of training data as context
    x_last30 = x_tr[:, -30:, :]
    c_last30 = c_tr[:, -30:, :]
    # Concatenate the last 23 days of context with the 7 days of eval data
    x_win = np.concatenate([x_last30[:, -23:, :], x_ev], axis=1)
    c_win = np.concatenate([c_last30[:, -23:, :], c_ev], axis=1)

    # Assemble the final input tensor for the model
    time_feature = np.broadcast_to(np.arange(16)[None, None, :, None] / 15.0, x_win.shape + (1,))
    cov_feature = np.broadcast_to(c_win[:, :, None, :], x_win.shape + (4,))
    X_eval_input = np.concatenate([x_win[..., None], cov_feature, time_feature], axis=-1)
    X_eval_input = X_eval_input.reshape(series_num, 30 * 16, 6)
    print(f"[RECOVERY DEBUG] Shape of final X_eval_input for prediction: {X_eval_input.shape}")

    # Predict using the already-fitted model
    print("[RECOVERY] Predicting on the constructed eval window...")
    prediction_results = model.predict({'X': X_eval_input})
    imputation = prediction_results['imputation']

    if len(imputation.shape) == 4:  # Handle multi-step imputation if present
        imputation = imputation.mean(axis=1)
    
    # Reshape imputation results and select the part corresponding to the eval set (last 7 days)
    imputed_sales_full_window = np.where(imputation > 0, imputation, 0)[..., 0].reshape(series_num, 30, 16)
    imputed_sales_eval_16h = imputed_sales_full_window[:, -7:, :]
    print(f"[RECOVERY DEBUG] Shape of imputed sales for eval set (16h): {imputed_sales_eval_16h.shape}")

    # Reconstruct the 24-hour daily sales for the eval set
    hrs_sale_eval_pred_24h = hrs_sale_eval_origin_24h.copy()
    hrs_sale_eval_pred_24h[:, :, 6:22] = imputed_sales_eval_16h
    sale_pred_eval_daily = hrs_sale_eval_pred_24h.sum(axis=-1).reshape(-1)

    # Create the output DataFrame
    output_df = df_ev.copy()
    output_df['sale_amount_pred_before_adjust'] = sale_pred_eval_daily
    # Ensure predicted demand is not less than observed sales (a common sense adjustment)
    output_df['sale_amount_pred'] = output_df[['sale_amount_pred_before_adjust', 'sale_amount']].max(axis=1)

    # Save the recovered eval demand
    output_dir = './demand'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f'demand_eval_{CONFIG["model"]}.parquet'
    output_df.to_parquet(os.path.join(output_dir, output_filename), index=False)
    print(f"[RECOVERY] ✅ Recovered EVAL set demand saved to: {os.path.join(output_dir, output_filename)}")

    # --- Evaluate recovery performance on the eval set ---
    print("\n--- Evaluating Recovery on Eval Set ---")
    all_metrics = []
    for subset_name, query in [
        ("All Eval", "stock_hour6_22_cnt >= 0"),
        ("Non-Stockout", "stock_hour6_22_cnt == 0"),
        ("Stockout", "stock_hour6_22_cnt > 0"),
    ]:
        subset_df = output_df.query(query)
        if subset_df.empty: continue

        den = subset_df['sale_amount'].sum() + 1e-9 # Add epsilon for safety

        # --- Metrics for model's raw prediction (before adjustment) ---
        wape_before = (subset_df['sale_amount_pred_before_adjust'].sub(subset_df['sale_amount']).abs().sum()) / den
        wpe_before  = (subset_df['sale_amount_pred_before_adjust'].sub(subset_df['sale_amount']).sum()) / den
        lift_ratio_before = (subset_df['sale_amount_pred_before_adjust'] >= subset_df['sale_amount']).mean()

        # --- Metrics for final prediction (after .max() adjustment) ---
        wape_after = (subset_df['sale_amount_pred'].sub(subset_df['sale_amount']).abs().sum()) / den
        wpe_after  = (subset_df['sale_amount_pred'].sub(subset_df['sale_amount']).sum()) / den
        lift_ratio_after = (subset_df['sale_amount_pred'] >= subset_df['sale_amount']).mean()

        metric_df = pd.DataFrame({
            'subset': [subset_name], 
            'wape_before_adjust': [wape_before], 'wpe_before_adjust': [wpe_before], 'lift_ratio_before_adjust': [lift_ratio_before],
            'wape_after_adjust': [wape_after], 'wpe_after_adjust': [wpe_after], 'lift_ratio_after_adjust': [lift_ratio_after]
        })
        all_metrics.append(metric_df)

    if all_metrics:
        final_metrics_df = pd.concat(all_metrics, ignore_index=True)
        print(final_metrics_df)
        metrics_filename = f'eval_recovery_metrics_{CONFIG["model"]}.csv'
        final_metrics_df.to_csv(os.path.join(output_dir, metrics_filename), index=False)
        print(f"[RECOVERY] ✅ Eval recovery metrics saved to: {os.path.join(output_dir, metrics_filename)}")

    # --- Gemini Code Assist: Calculate decoupling score for the eval set ---
    evaluation_decoupling(output_df, CONFIG, data_split='eval')
    
    # --- Gemini Code Assist: Calculate non-stockout consistency for the eval set ---
    evaluation_non_stockout_consistency(output_df, CONFIG, data_split='eval')

def evaluation_mnar(X, imputation, CONFIG):
    # --- Gemini Code Assist: Load dataset from local disk ---
    dataset = load_from_disk("../../frn_50k_local_dataset")
    # ------------------------------------------------------
    data = dataset['train'].to_pandas()
    data = data.sort_values(by=['store_id', 'product_id', 'dt'])
    horizon=90
    series_num = data.shape[0]//horizon
    hours_sale = np.array(data['hours_sale'].tolist())
    hours_stock_status = np.array(data['hours_stock_status'].tolist())

    hours_sale_origin = hours_sale.reshape(series_num*3, 30, 24)

    stock_hour_X = np.isnan(X[...,0].reshape(-1,30,16)).sum(axis=-1).reshape(-1,90).reshape(-1)
    stock_hour_origin = hours_stock_status[:,6:22].sum(axis=1)
    hours_sale_impute = hours_sale_origin.copy()
    hours_sale_impute[...,6:22] = imputation.reshape(-1, 30, 16)
    sale_amount_pred = hours_sale_impute.sum(axis=-1).reshape(-1, 90).reshape(-1)
    valid_idx = (stock_hour_X>0)&(stock_hour_origin==0) #&(data['mu']>=1)
    sale_amount = data['sale_amount'].values
    wape = (np.abs(sale_amount_pred-sale_amount)*valid_idx).sum()/(sale_amount*valid_idx).sum()
    wpe = ((sale_amount_pred-sale_amount)*valid_idx).sum()/(sale_amount*valid_idx).sum()

    # --- Gemini Code Assist: Saving metrics to file ---
    print(f"MNAR Evaluation for {CONFIG['model']} with missing_rate={CONFIG['missing_rate']}:")
    print(f"  WAPE: {wape}")
    print(f"  WPE: {wpe}")

    results_df = pd.DataFrame({
        'model': [CONFIG['model']],
        'missing_rate': [CONFIG['missing_rate']],
        'wape': [wape],
        'wpe': [wpe],
        'timestamp': [datetime.now()]
    })
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'mnar_evaluation_metrics.csv')
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, mode='w', header=True, index=False)
    
def evaluation_decoupling(data, CONFIG, data_split: str):
    print(f"\n--- Calculating Decoupling Score for {data_split.upper()} set ---")
    # --- Gemini Code Assist: Use both before/after adjustment predictions for a complete analysis ---
    cols_to_use = ['city_id', 'store_id', 'product_id', 'dt', 'holiday_flag', 'discount', 'sale_amount', 'stock_hour6_22_cnt']
    pred_cols = ['sale_amount_pred']
    if 'sale_amount_pred_before_adjust' in data.columns:
        pred_cols.append('sale_amount_pred_before_adjust')
    
    df = data[cols_to_use + pred_cols].copy()
    mu = df.query('stock_hour6_22_cnt==0').groupby(['store_id', 'product_id'])['sale_amount'].mean()
    mu = mu.reset_index().rename(columns={'sale_amount':'mu'})
    corr = df.query('stock_hour6_22_cnt>0').groupby(['store_id', 'product_id', 'holiday_flag']).apply(lambda subdf:subdf[['stock_hour6_22_cnt', 'sale_amount'] + pred_cols].corr().iloc[:1, 1:])
    stock_nunique = df.query('stock_hour6_22_cnt>0').groupby(['store_id', 'product_id', 'holiday_flag']).agg({'stock_hour6_22_cnt':'nunique'}).reset_index()
    stock_nunique = stock_nunique.rename(columns={'stock_hour6_22_cnt':'nunique'})
    corr = corr.reset_index().merge(mu, on=['store_id', 'product_id']).merge(stock_nunique.query('nunique>3'), on=['store_id', 'product_id', 'holiday_flag'])
    metric = pd.DataFrame({
            'method': ['sale_amount'] + pred_cols,
            'decoupling score':np.nansum(corr[['sale_amount'] + pred_cols].values * corr[['mu']].values, axis=0)/corr['mu'].sum()
             })
    print(metric)

    # --- Gemini Code Assist: Saving metrics to file ---
    metric['model'] = CONFIG['model']
    metric['timestamp'] = datetime.now()
    metric['data_split'] = data_split # Add the data split information
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'decoupling_scores.csv')
    if os.path.exists(results_file):
        metric.to_csv(results_file, mode='a', header=False, index=False)
    else:
        metric.to_csv(results_file, mode='w', header=True, index=False)

def evaluation_non_stockout_consistency(data, CONFIG, data_split: str):
    """
    分析在非缺货日，恢复后的需求与原始销量之间的差异。
    """
    print(f"\n--- Calculating Non-Stockout Consistency for {data_split.upper()} set ---")
    non_stockout_df = data[data['stock_hour6_22_cnt'] == 0].copy()

    if non_stockout_df.empty:
        print("No non-stockout days found to analyze.")
        return

    # 使用调整前的预测值进行比较，以反映模型真实重构能力
    pred_col = 'sale_amount_pred_before_adjust' if 'sale_amount_pred_before_adjust' in non_stockout_df.columns else 'sale_amount_pred'
    
    diff = (non_stockout_df[pred_col] - non_stockout_df['sale_amount']).abs()
    num_different = (diff > 1e-6).sum()
    
    denominator = non_stockout_df['sale_amount'].sum()
    wape = np.nan
    wpe = np.nan
    if denominator > 0:
        wape = diff.sum() / denominator
        wpe = (non_stockout_df[pred_col] - non_stockout_df['sale_amount']).sum() / denominator

    metric_df = pd.DataFrame({
        'model': [CONFIG['model']],
        'data_split': [data_split],
        'wape': [wape],
        'wpe': [wpe],
        'num_samples': [len(non_stockout_df)],
        'num_different_rows': [num_different],
        'timestamp': [datetime.now()]
    })
    print(metric_df[['model', 'data_split', 'wape', 'wpe', 'num_different_rows']].round(6))

    results_dir = './results'
    # --- Gemini Code Assist: Use a unique filename for each model to avoid race conditions ---
    # This is a safer approach in parallel environments than appending to a single file.
    model_name = CONFIG['model']
    results_file = os.path.join(results_dir, f'non_stockout_consistency_metrics_{model_name}.csv')
    metric_df.to_csv(results_file, mode='w', header=True, index=False)
    
    
# default params config
CONFIG = {
    'model': 'DLinear',
    'saving_path': './save',
    'EPOCHS': 5,
    'batch_size': 128,
    'patience': 5,
    'n_layers': 2,
    'd_model': 64,
    'd_ffn': 32,
    'n_heads': 4,
    'd_k': 16,
    'd_v': 16,
    'dropout': 0.,
    'attn_dropout': 0.,
    'lr': 0.001, 
    'weight_decay': 1e-5, 
    'OT': 1,
    'missing_rate':0.3,
    'n_patches':7,
    'alpha': 1e-2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default='TimesNet',
        help="Demand Recovery Model, default='TimesNet'"
    )
    parser.add_argument(
        "--missing_rate", 
        type=float, 
        default=0.,
        help="Missing Rate for Artificial MNAR Evaluation, default missing_rate = 0 for latent demand recovery"
    )
    args = parser.parse_args()
    print(args)
    CONFIG['model'] = args.model
    CONFIG['missing_rate'] = args.missing_rate
    ## set random seed
    set_seed(seed_value=1024)
    ## latent demand recovery
    demand_df = demand_recovery(CONFIG)
