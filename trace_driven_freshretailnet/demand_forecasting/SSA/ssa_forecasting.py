import pandas as pd
import numpy as np
from datasets import load_from_disk
import os
import argparse

def ssa_predict(demand_df, target_col='sale_amount', recovery_model_name='DLinear'):
    # --- Gemini Code Assist: Load dataset from local file ---
    dataset = load_from_disk("../../frn_50k_local_dataset")
    # ---------------------------------------------------------
    eval_df = dataset['eval'].to_pandas()

    # --- Gemini Code Assist: Ensure consistent datetime type before concat/sort ---
    demand_df['dt'] = pd.to_datetime(demand_df['dt'])
    eval_df['dt'] = pd.to_datetime(eval_df['dt'])
    # all_data = pd.concat([demand_df, eval_df], axis=0).sort_values(by=['store_id', 'product_id', 'dt'])
    all_data = pd.concat([demand_df, eval_df], axis=0)

    # calendar feature
    all_data['time_idx'] = (all_data['dt'] - all_data['dt'].min()).dt.days
    all_data['dow'] = all_data['dt'].dt.dayofweek
    all_data.loc[all_data['dt'].isin(['2024-05-02','2024-05-03','2024-05-04']), 'holiday_flag']=0
    all_data.loc[all_data['holiday_flag']==1, 'dow'] = 5
    all_data.loc[(all_data['holiday_flag']==0)&(all_data['dow'].isin([5,6])), 'dow'] = 0
    # precipitation feature
    all_data['is_precpt'] = (all_data['precpt']>3.5).astype(int)
    # continuouse and catigorical variables
    ### 2601: add to make sure ordering stable and not product different results in two runs
    all_data = all_data.sort_values(
        by=['store_id', 'product_id', 'dt'],
        kind='mergesort'   # stable sort
    ).reset_index(drop=True)

    # sanity check: every series must have exactly 97 rows
    counts = all_data.groupby(['store_id','product_id']).size()
    bad = counts[counts != 97]
    if not bad.empty:
        raise ValueError(f"Found series with non-97 length. Example:\n{bad.head()}")
    
    x_cont = all_data[['time_idx', 'discount', target_col]].values.reshape(-1, 97, 3)
    x_cat = all_data[['dow', 'holiday_flag', 'is_precpt']].values.reshape(-1, 97, 3)
    # config
    max_encoder_length=90
    time_idx=0
    discount_idx=1
    dow_idx=0
    holiday_idx=1
    is_precpt_idx=2
    holiday_label = 5
    # weight
    date_distance = 18 - x_cont[:, :max_encoder_length, time_idx][:,None,:]/5
    discount_distance = 1 - np.abs((x_cont[:, max_encoder_length:, discount_idx][...,None] - x_cont[:, :max_encoder_length, discount_idx][:,None,:]))
    decoder_dow = (x_cat[:, max_encoder_length:, dow_idx][...,None])
    encoder_dow = (x_cat[:, :max_encoder_length, dow_idx][:,None,:])
    dow_distance = decoder_dow - encoder_dow
    decoder_precpt = (x_cat[:, max_encoder_length:, is_precpt_idx][...,None])
    encoder_precpt = (x_cat[:, :max_encoder_length, is_precpt_idx][:,None,:])
    precpt_distance = decoder_precpt - encoder_precpt
    date_distance = np.where((decoder_dow == holiday_label).astype(int) - (encoder_dow == holiday_label).astype(int) == 0, date_distance - 3, date_distance)
    date_distance = np.where((decoder_dow != holiday_label) & (encoder_dow == holiday_label), date_distance + 3, date_distance)
    date_distance = np.where(dow_distance == 0, date_distance - 3, date_distance)
    date_distance = np.where((precpt_distance == 0) & (decoder_precpt == 1), date_distance - 3, date_distance)
    date_distance = np.where(precpt_distance != 0, date_distance + 3, date_distance)
    date_distance = np.clip(date_distance, a_min=1, a_max=18)
    weight = (18-date_distance) * discount_distance**2
    weight = np.exp(weight)/np.exp(weight).sum(axis=-1, keepdims=True)

    # high sale & low sale info
    mu = all_data.query("dt<='2024-06-25'").groupby(['store_id', 'product_id']).agg({'sale_amount':'mean'})
    mu = mu.reset_index().rename(columns={'sale_amount':'psd'})
    all_data = all_data.merge(mu, on=['store_id', 'product_id'], how='left')

    pdf = all_data.query("dt>='2024-06-26'").copy()
    target_idx = 2
    target = x_cont[:, :max_encoder_length, target_idx:target_idx+1]
    index = encoder_dow.squeeze()[...,None] == np.arange(6)
    mean_sale = np.nanmean(np.where(index, target, np.nan), axis=1) + 0.001 # noqa
    season_ratio = np.nanmedian(mean_sale[...,None] / mean_sale[:,None,...], axis=0)
    ratio = season_ratio[decoder_dow, encoder_dow]
    pred = (weight * ratio)@target
    pdf['sale_amount_pred'] = pred.reshape(-1)
    # overall
    metric = pd.concat([
        evaluation(pdf, 'psd>=0', target_col), # overall
        evaluation(pdf, 'psd<1', target_col), # low sale
        evaluation(pdf, 'psd>=1', target_col), # high sale
    ], axis=0)
    print(metric)
    run_type = "recovered" if "pred" in target_col else "censored"

    # --- Gemini Code Assist: Saving prediction results ---
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    if run_type == "censored":
        output_filename = 'SSA_predictions_censored.parquet'
    else:   
        output_filename = f'SSA_predictions_{run_type}_by_{recovery_model_name}.parquet'
    pdf.to_parquet(os.path.join(results_dir, output_filename))
    print(f"SSA predictions (without sigma) saved to {os.path.join(results_dir, output_filename)}")
    
    # --- Per user request, only print non-stockout metrics, do not save them. ---
    print("\n--- Metrics on Non-Stockout Days (for reference only) ---")
    print(metric)
    
    # --- Gemini Code Assist: Estimate residual distribution per PSD group ---
    print("\n--- Estimating Residual Distribution ---")
    # Calculate residuals on non-stockout days
    non_stockout_df = pdf.query('stock_hour6_22_cnt == 0').copy()
    non_stockout_df['residual'] = non_stockout_df['sale_amount'] - non_stockout_df['sale_amount_pred']

    # Calculate standard deviation of residuals for each PSD group
    residual_std = non_stockout_df.groupby(non_stockout_df['psd'] >= 1)['residual'].std().to_dict()
    sigma_low_psd = residual_std.get(False, 0)  # PSD < 1
    sigma_high_psd = residual_std.get(True, 0)   # PSD >= 1

    print(f"Std Dev of Residuals (PSD < 1): {sigma_low_psd:.4f}")
    print(f"Std Dev of Residuals (PSD >= 1): {sigma_high_psd:.4f}")

    # Add the estimated sigma to each row in the prediction dataframe
    pdf['pred_sigma'] = np.where(pdf['psd'] < 1, sigma_low_psd, sigma_high_psd)
    
    # Re-save the prediction file with the new 'pred_sigma' column
    pdf.to_parquet(os.path.join(results_dir, output_filename))
    print(f"SSA predictions with 'pred_sigma' saved to {os.path.join(results_dir, output_filename)}")
    # --------------------------------------------------------------------

    return pdf
    
def evaluation(pdf, condition='psd>=0', target_col='sale_amount'):
    res = []
    pred_col = 'sale_amount_pred'
    wape_list,mae_list,wpe_list = [],[],[]
    for target_dt, subdf in pdf.query('stock_hour6_22_cnt==0').query(condition).groupby('dt'):
        mae = (subdf['sale_amount'] - subdf[pred_col]).abs().sum()
        wape = mae / subdf['sale_amount'].sum()
        wape_list.append(wape)
        mae_list.append((subdf['sale_amount'] - subdf[pred_col]).abs().mean())
        wpe_list.append((subdf[pred_col]-subdf['sale_amount']).sum()/subdf['sale_amount'].sum())
    res.append(pd.DataFrame({'demand':[pred_col], 'wape':[round(np.mean(wape_list),4)], 'wpe':[round(np.mean(wpe_list),4)], 'mae':[round(np.mean(mae_list),4)]}))
    metric = pd.concat(res)
    metric['group'] = condition
    metric = metric[['group', 'demand', 'wape', 'wpe', 'mae']]
    return metric

def evaluation_vs_recovered(pdf, ground_truth_path, pred_col_name, data_type, is_train_set=False):
    """Compares forecasting predictions against recovered demand from the eval set."""
    split_name = "TRAIN" if is_train_set else "EVAL"
    print(f"\n--- Evaluating Forecast vs. Recovered Demand (on {split_name} Set) ---")
    try:
        ground_truth_df = pd.read_parquet(ground_truth_path)
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {ground_truth_path}. Skipping this evaluation.")
        return

    # --- Gemini Code Assist: Fix merge error by ensuring consistent key types ---
    # Convert 'dt' columns in both dataframes to string format 'YYYY-MM-DD' before merging.
    # This prevents the "You are trying to merge on datetime64[ns] and object columns" error.
    pdf['dt'] = pd.to_datetime(pdf['dt']).dt.strftime('%Y-%m-%d')
    ground_truth_df['dt'] = pd.to_datetime(ground_truth_df['dt']).dt.strftime('%Y-%m-%d')
    ground_truth_df = ground_truth_df.rename(columns={'sale_amount_pred': 'recovered_demand_truth'})   ## recover的时候recover结果存在sale_amount_pred列 在这里和ssa预测结果列重名 先换个名字

    # Merge forecast predictions with recovered demand ground truth
    merged_df = pd.merge(
        pdf[['store_id', 'product_id', 'dt', 'stock_hour6_22_cnt', pred_col_name]],
        ground_truth_df[['store_id', 'product_id', 'dt', 'recovered_demand_truth']],
        on=['store_id', 'product_id', 'dt']
    )

    all_metrics = []
    for subset_name, query in [
        ("All Eval", "stock_hour6_22_cnt >= 0"),
        ("Non-Stockout", "stock_hour6_22_cnt == 0"),
        ("Stockout", "stock_hour6_22_cnt > 0"),
    ]:
        subset_df = merged_df.query(query)
        if subset_df.empty: continue
        wape = (subset_df[pred_col_name] - subset_df['recovered_demand_truth']).abs().sum() / (subset_df['recovered_demand_truth'].sum() + 1e-8)
        wpe = (subset_df[pred_col_name] - subset_df['recovered_demand_truth']).sum() / (subset_df['recovered_demand_truth'].sum() + 1e-8)
        me = (subset_df[pred_col_name] - subset_df['recovered_demand_truth']).mean()
        all_metrics.append({'subset': subset_name, 'wape': round(wape, 4), 'wpe': round(wpe, 4), 'me': round(me, 4)})

    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n--- Metrics on {split_name} Set ---")
    print(metrics_df)

    # --- Gemini Code Assist: Save evaluation metrics to a CSV file with the requested naming convention ---
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    eval_model_name = os.path.basename(ground_truth_path).replace('demand_eval_', '').replace('demand_', '').replace('.parquet', '')
    
    train_str = "train_" if is_train_set else ""
    if 'censored' in data_type:
        metrics_filename = f"SSA_{train_str}metrics_censored_by_{eval_model_name}.csv"
    else:
        recovery_model = data_type.replace('recovered_by_', '')
        metrics_filename = f"SSA_{train_str}metrics_recovered_by_{recovery_model}.csv"

    metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
    print(f"SSA evaluation metrics saved to {os.path.join(results_dir, metrics_filename)}")

def resolve_eval_ground_truth_path(args):
    if args.recovered_eval_path:
        return args.recovered_eval_path
    return f'../../latent_demand_recovery/exp/demand/demand_eval_{args.recovery_model_name}.parquet'

def resolve_train_ground_truth_path(args, eval_ground_truth_path):
    if args.demand:
        return args.demand_path
    if eval_ground_truth_path and 'demand_eval_' in eval_ground_truth_path:
        return eval_ground_truth_path.replace('demand_eval_', 'demand_')
    return f'../../latent_demand_recovery/exp/demand/demand_{args.recovery_model_name}.parquet'

def predict_and_evaluate_on_train_split(demand_df, target_col, args, data_type):
    """
    Predicts only the last 7 days of the training set (days 84-90) and evaluates them.
    """
    print("\n--- Starting Prediction and Evaluation on TRAINING Set (Last 7 Days) ---")
    
    # 1. Prepare data
    all_data = demand_df.sort_values(by=['store_id', 'product_id', 'dt']).copy()
    all_data['dt'] = pd.to_datetime(all_data['dt']) # Ensure datetime
    all_data['time_idx'] = (all_data['dt'] - all_data['dt'].min()).dt.days
    all_data['dow'] = all_data['dt'].dt.dayofweek
    all_data.loc[all_data['dt'].isin(['2024-05-02','2024-05-03','2024-05-04']), 'holiday_flag']=0
    all_data.loc[all_data['holiday_flag']==1, 'dow'] = 5
    all_data.loc[(all_data['holiday_flag']==0)&(all_data['dow'].isin([5,6])), 'dow'] = 0
    all_data['is_precpt'] = (all_data['precpt']>3.5).astype(int)

    # 2601
    if len(all_data) % 90 != 0:
        raise ValueError(
            f"Train data rows {len(all_data)} not divisible by 90. "
            "Check missing days / ordering."
        )
    
    x_cont = all_data[['time_idx', 'discount', target_col]].values
    x_cat = all_data[['dow', 'holiday_flag', 'is_precpt']].values

    num_series = len(all_data) // 90
    x_cont = x_cont.reshape(num_series, 90, 3)
    x_cat = x_cat.reshape(num_series, 90, 3)

    # 2. Use the final sliding window (history days 15-83, predict 84-90)
    start_idx = 14
    encoder_end = 69 + start_idx   # 83
    decoder_end = 76 + start_idx   # 90

    encoder_cont = x_cont[:, start_idx:encoder_end, :]
    decoder_cont_features = x_cont[:, encoder_end:decoder_end, :]
    encoder_cat = x_cat[:, start_idx:encoder_end, :]
    decoder_cat = x_cat[:, encoder_end:decoder_end, :]

    # Apply SSA weighting logic
    date_distance = 18 - encoder_cont[:, :, 0][:, None, :] / 5
    discount_distance = 1 - np.abs(decoder_cont_features[:, :, 1][..., None] - encoder_cont[:, :, 1][:, None, :])
    dow_distance = decoder_cat[:, :, 0][..., None] - encoder_cat[:, :, 0][:, None, :]

    weight = (18 - date_distance) * discount_distance**2
    weight = np.exp(weight) / np.exp(weight).sum(axis=-1, keepdims=True)

    target = encoder_cont[:, :, 2:3]
    season_ratio = np.ones((6, 6))
    ratio = season_ratio[
        decoder_cat[:, :, 0, None],
        encoder_cat[:, :, 0, None].transpose(0, 2, 1)
    ]

    pred = (weight * ratio) @ target
    final_preds = pred.reshape(-1)

    # 3. Assemble dataframe for last 7 days
    df_train_last_7 = all_data.groupby(['store_id', 'product_id']).tail(7).copy()
    df_train_last_7['sale_amount_pred'] = final_preds

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    if 'censored' in data_type:
        train_pred_filename = 'SSA_train_predictions_censored.parquet'
    else:
        recovery_model = data_type.replace('recovered_by_', '')
        train_pred_filename = f'SSA_train_predictions_recovered_by_{recovery_model}.parquet'
    df_train_last_7.to_parquet(os.path.join(results_dir, train_pred_filename), index=False)
    print(f"SSA training set predictions (last 7 days) saved to {os.path.join(results_dir, train_pred_filename)}")
    return df_train_last_7
    # -------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demand_path", 
        type=str, 
        default='../../latent_demand_recovery/exp/demand/demand.parquet',
        help="demand data path, default '../../latent_demand_recovery/exp/demand/demand.parquet'"
    )
    parser.add_argument(
        "--demand",
        action='store_true',
        help="use recoverd demand or not"
    )
    parser.add_argument(
        "--recovered_eval_path",
        type=str,
        default=None,
        help="Path to the recovered demand file for the eval set."
    )
    parser.add_argument(
        "--recovery_model_name",
        type=str,
        default="DLinear",
        help="Name of the recovery model used, for file naming."
    )
    parser.add_argument(
        "--skip_predict",
        action='store_true',
        help="Skip the prediction step and only run evaluations."
    )
    args = parser.parse_args()

    recover_or_censor = True if args.demand else False
    
    # --- Gemini Code Assist: Implement --skip_predict logic ---
    # If skip_predict is True, we only perform evaluation against a recovered demand set.
    if args.skip_predict:
        if not args.recovered_eval_path:
            print("Error: --skip_predict requires --recovered_eval_path to be set.")
            exit()

        print("\n--- Running in Evaluation-Only Mode (skip_predict=True) ---")
        results_dir = './results'
        
        # --- Gemini Code Assist: Determine data_type for correct metric file naming ---
        if not args.demand:
            data_type = 'censored'
            prediction_filename = 'SSA_predictions_censored.parquet'
            train_prediction_filename = 'SSA_train_predictions_censored.parquet'
        else:
            data_type = f'recovered_by_{args.recovery_model_name}'
            prediction_filename = f'SSA_predictions_recovered_by_{args.recovery_model_name}.parquet'
            train_prediction_filename = f'SSA_train_predictions_recovered_by_{args.recovery_model_name}.parquet'
        
        train_pred_file_path = os.path.join(results_dir, train_prediction_filename)
        print(f"Loading existing predictions from: {train_pred_file_path}")
        train_df_pred = pd.read_parquet(train_pred_file_path)
        eval_ground_truth_path = args.recovered_eval_path
        train_ground_truth_path = resolve_train_ground_truth_path(args, eval_ground_truth_path)

        if train_ground_truth_path and os.path.exists(train_ground_truth_path):
            evaluation_vs_recovered(train_df_pred, train_ground_truth_path, 'sale_amount_pred', data_type, is_train_set=True)
        else:
            print(f"Warning: Training ground truth file not found at '{train_ground_truth_path}'. Skipping train set evaluation.")

        pred_file_path = os.path.join(results_dir, prediction_filename)
        print(f"Loading existing predictions from: {pred_file_path}")
        df_pred = pd.read_parquet(pred_file_path)

        evaluation_vs_recovered(df_pred, eval_ground_truth_path, 'sale_amount_pred', data_type, is_train_set=False)
        exit() # End the script after evaluation

    if args.demand:
        data_type = f'recovered_by_{args.recovery_model_name}'
        target_col = 'sale_amount_pred'
        demand_df = pd.read_parquet(args.demand_path)
    else:
        target_col = 'sale_amount'
        # --- Gemini Code Assist: Load dataset from local file ---
        dataset = load_from_disk("../../frn_50k_local_dataset")
        # ---------------------------------------------------------
        demand_df = dataset['train'].to_pandas()
        data_type = 'censored'

    eval_ground_truth_path = resolve_eval_ground_truth_path(args)
    train_ground_truth_path = resolve_train_ground_truth_path(args, eval_ground_truth_path)

    # --- Run prediction and evaluation on the training set ---
    df_train_last_21 = predict_and_evaluate_on_train_split(demand_df.copy(), target_col, args, data_type)
    evaluation_vs_recovered(df_train_last_21, train_ground_truth_path, 'sale_amount_pred', data_type, is_train_set=True)

    # --- Run prediction and evaluation on the evaluation set ---
    pdf = ssa_predict(demand_df, target_col, recovery_model_name=args.recovery_model_name)
    evaluation_vs_recovered(pdf, eval_ground_truth_path, 'sale_amount_pred', data_type, is_train_set=False)
