import pandas as pd
import numpy as np
from datasets import load_from_disk
import os
import argparse

def seasonal_naive_predict(demand_df, target_col='sale_amount', recovery_model_name='DLinear'):
    """
    使用季节性朴素预测（按星期几的均值）来生成预测。
    """
    # 1. 加载评估集并准备数据
    dataset = load_from_disk("../../frn_50k_local_dataset")
    eval_df = dataset['eval'].to_pandas()

    demand_df['dt'] = pd.to_datetime(demand_df['dt'])
    eval_df['dt'] = pd.to_datetime(eval_df['dt'])
    
    # 2. 计算核心的“按星期几的平均需求”查找表
    # 使用训练数据(demand_df)来构建模型
    demand_df['dow'] = demand_df['dt'].dt.dayofweek
    # 计算每个 (store_id, product_id, dow) 组合的平均销售额
    dow_mean_map = demand_df.groupby(['store_id', 'product_id', 'dow'])[target_col].mean().to_frame('sale_amount_pred').reset_index()

    # 3. 为评估集生成预测
    eval_df['dow'] = eval_df['dt'].dt.dayofweek
    # 将历史均值合并到评估集上，作为预测值
    pdf = pd.merge(eval_df, dow_mean_map, on=['store_id', 'product_id', 'dow'], how='left')
    
    # 对于在训练集中未出现过的(item, dow)组合，用该item的总体平均值填充
    item_mean_map = demand_df.groupby(['store_id', 'product_id'])[target_col].mean().to_frame('item_mean')
    pdf = pd.merge(pdf, item_mean_map, on=['store_id', 'product_id'], how='left')
    pdf['sale_amount_pred'].fillna(pdf['item_mean'], inplace=True)
    pdf.fillna(0, inplace=True) # 对于在训练集中完全未见过的item，预测为0

    # 4. 计算PSD（用于后续分组评估和sigma估计）
    mu = demand_df.groupby(['store_id', 'product_id'])['sale_amount'].mean().to_frame('psd')
    pdf = pdf.merge(mu, on=['store_id', 'product_id'], how='left')

    # 5. 保存预测结果并估算sigma（与其它模型保持一致）
    run_type = "recovered" if "pred" in target_col else "censored"
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    if run_type == "censored":
        output_filename = 'SeasonalNaive_predictions_censored.parquet'
    else:   
        output_filename = f'SeasonalNaive_predictions_{run_type}_by_{recovery_model_name}.parquet'
    
    # --- 估算残差标准差 (sigma) ---
    print("\n--- Estimating Residual Distribution ---")
    # 在非缺货日计算残差
    non_stockout_df = pdf.query('stock_hour6_22_cnt == 0').copy()
    non_stockout_df['residual'] = non_stockout_df['sale_amount'] - non_stockout_df['sale_amount_pred']

    # 按PSD分组计算残差的标准差
    residual_std = non_stockout_df.groupby(non_stockout_df['psd'] >= 1)['residual'].std().to_dict()
    sigma_low_psd = residual_std.get(False, 0)
    sigma_high_psd = residual_std.get(True, 0)

    print(f"Std Dev of Residuals (PSD < 1): {sigma_low_psd:.4f}")
    print(f"Std Dev of Residuals (PSD >= 1): {sigma_high_psd:.4f}")

    # 将估算的sigma添加到预测DataFrame中
    pdf['pred_sigma'] = np.where(pdf['psd'] < 1, sigma_low_psd, sigma_high_psd)
    
    # 保存带有sigma的完整预测文件
    pdf.to_parquet(os.path.join(results_dir, output_filename))
    print(f"SeasonalNaive predictions with 'pred_sigma' saved to {os.path.join(results_dir, output_filename)}")

    return pdf

def evaluation_vs_recovered(pdf, ground_truth_path, pred_col_name, data_type, is_train_set=False):
    """
    将预测结果与恢复后的“真实需求”进行比较，计算并保存评估指标。
    此函数与其它模型中的版本完全相同，以确保评估标准统一。
    """
    split_name = "TRAIN" if is_train_set else "EVAL"
    print(f"\n--- Evaluating Forecast vs. Recovered Demand (on {split_name} Set) ---")
    try:
        ground_truth_df = pd.read_parquet(ground_truth_path)
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {ground_truth_path}. Skipping this evaluation.")
        return

    pdf['dt'] = pd.to_datetime(pdf['dt']).dt.strftime('%Y-%m-%d')
    ground_truth_df['dt'] = pd.to_datetime(ground_truth_df['dt']).dt.strftime('%Y-%m-%d')
    ground_truth_df = ground_truth_df.rename(columns={'sale_amount_pred': 'recovered_demand_truth'})

    merged_df = pd.merge(
        pdf,
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
        denominator = subset_df['recovered_demand_truth'].sum() + 1e-8
        wape = (subset_df[pred_col_name] - subset_df['recovered_demand_truth']).abs().sum() / denominator
        wpe = (subset_df[pred_col_name] - subset_df['recovered_demand_truth']).sum() / denominator
        me = (subset_df[pred_col_name] - subset_df['recovered_demand_truth']).mean()
        all_metrics.append({'subset': subset_name, 'wape': round(wape, 4), 'wpe': round(wpe, 4), 'me': round(me, 4)})

    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n--- Metrics on {split_name} Set ---")
    print(metrics_df)

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    eval_model_name = os.path.basename(ground_truth_path).replace('demand_eval_', '').replace('demand_', '').replace('.parquet', '')
    
    train_str = "train_" if is_train_set else ""
    if 'censored' in data_type:
        metrics_filename = f"SeasonalNaive_{train_str}metrics_censored_by_{eval_model_name}.csv"
    else:
        recovery_model = data_type.replace('recovered_by_', '')
        metrics_filename = f"SeasonalNaive_{train_str}metrics_recovered_by_{recovery_model}.csv"

    metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
    print(f"SeasonalNaive evaluation metrics saved to {os.path.join(results_dir, metrics_filename)}")

def predict_and_evaluate_on_train_split(demand_df, target_col, data_type):
    """
    使用训练集全量历史构建季节均值，但只对最后 7 天进行预测评估。
    """
    print("\n--- Starting Prediction and Evaluation on TRAINING Set (Last 7 Days) ---")
    
    df_train_all = demand_df.copy()
    df_train_all['dt'] = pd.to_datetime(df_train_all['dt'])
    df_train_all = df_train_all.sort_values(by=['store_id', 'product_id', 'dt'])
    df_train_all['dow'] = df_train_all['dt'].dt.dayofweek

    # 1. Correctly split data to prevent leakage: use all but last 7 days for training
    df_train_tail = df_train_all.groupby(['store_id', 'product_id']).tail(7).copy()
    df_train_set = df_train_all.drop(df_train_tail.index)

    # 2. Build (item, dow) mean lookup table from the TRAINING data.
    #    Rename the calculated column to 'dow_pred' to avoid merge conflicts.
    dow_mean_map = df_train_set.groupby(['store_id', 'product_id', 'dow'])[target_col].mean().to_frame('dow_pred').reset_index()

    # 3. Merge the day-of-week predictions into the test set (last 7 days).
    df_train_tail = pd.merge(df_train_tail, dow_mean_map, on=['store_id', 'product_id', 'dow'], how='left')

    # 4. For missing combinations, build a fallback using the overall item mean from TRAINING data.
    item_mean_map = df_train_set.groupby(['store_id', 'product_id'])[target_col].mean().to_frame('item_mean').reset_index()
    df_train_tail = pd.merge(df_train_tail, item_mean_map, on=['store_id', 'product_id'], how='left')
    
    # 5. Create the final prediction column, using the day-of-week prediction first.
    df_train_tail['sale_amount_pred'] = df_train_tail['dow_pred']
    #    Fill any missing values with the overall item mean, then with 0.
    df_train_tail['sale_amount_pred'].fillna(df_train_tail['item_mean'], inplace=True)
    df_train_tail.fillna(0, inplace=True)
    
    # 6. Clean up intermediate columns. The original rename is no longer needed.
    df_train_tail.drop(columns=['dow_pred', 'item_mean'], inplace=True)
    print(df_train_tail.info())

    # PSD 信息用于 residual 分组
    mu = df_train_all.groupby(['store_id', 'product_id'])['sale_amount'].mean().to_frame('psd')
    df_train_tail = df_train_tail.merge(mu, on=['store_id', 'product_id'], how='left')

    # 3. 保存最后 7 天训练预测
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    if 'censored' in data_type:
        train_pred_filename = 'SeasonalNaive_train_predictions_censored.parquet'
    else:
        recovery_model = data_type.replace('recovered_by_', '')
        train_pred_filename = f'SeasonalNaive_train_predictions_recovered_by_{recovery_model}.parquet'

    cols_to_save = ['store_id', 'product_id', 'dt', 'sale_amount_pred', 'stock_hour6_22_cnt', 'psd']
    df_train_tail[cols_to_save].to_parquet(os.path.join(results_dir, train_pred_filename), index=False)
    print(f"SeasonalNaive training set predictions (last 7 days) saved to {os.path.join(results_dir, train_pred_filename)}")
    
    return df_train_tail

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Seasonal Naive Forecasting Model")
    parser.add_argument("--demand_path", type=str, help="Path to the recovered demand data.")
    parser.add_argument("--demand", action='store_true', help="Flag to use recovered demand.")
    parser.add_argument("--recovered_eval_path", type=str, help="Path to the recovered demand for the eval set.")
    parser.add_argument("--recovery_model_name", type=str, default="DLinear", help="Name of the recovery model.")
    parser.add_argument("--train_ground_truth_path", type=str, help="Path to the recovered demand for the train set evaluation.")
    parser.add_argument("--skip_predict", action='store_true', help="Skip prediction and only run evaluation.")
    args = parser.parse_args()

    # --- 主逻辑 ---
    if args.skip_predict:
        # 如果跳过预测，则只执行评估
        if not args.recovered_eval_path:
            print("Warning: --recovered_eval_path not provided. Skipping evaluation on EVAL set.")

        print("\n--- Running in Evaluation-Only Mode (skip_predict=True) ---")
        results_dir = './results'
        
        data_type = f'recovered_by_{args.recovery_model_name}' if args.demand else 'censored'
        
        # 评估 Eval Set
        if args.recovered_eval_path and os.path.exists(args.recovered_eval_path):
            pred_filename = f"SeasonalNaive_predictions_{data_type}_by_{args.recovery_model_name}.parquet" if args.demand else "SeasonalNaive_predictions_censored.parquet"
            pred_file_path = os.path.join(results_dir, pred_filename)
            print(f"Loading existing EVAL predictions from: {pred_file_path}")
            df_pred = pd.read_parquet(pred_file_path)
            evaluation_vs_recovered(df_pred, args.recovered_eval_path, 'sale_amount_pred', data_type, is_train_set=False)

        # 评估 Train Set
        # --- Gemini Code Assist: Use the new --train_ground_truth_path argument ---
        ground_truth_path_train = args.train_ground_truth_path
        if ground_truth_path_train and os.path.exists(ground_truth_path_train):
            train_pred_filename = f"SeasonalNaive_train_predictions_{data_type}_by_{args.recovery_model_name}.parquet" if args.demand else "SeasonalNaive_train_predictions_censored.parquet"
            train_pred_file_path = os.path.join(results_dir, train_pred_filename)
            print(f"Loading existing TRAIN predictions from: {train_pred_file_path}")
            train_df_pred = pd.read_parquet(train_pred_file_path)
            evaluation_vs_recovered(train_df_pred, ground_truth_path_train, 'sale_amount_pred', data_type, is_train_set=True)
        else:
            print("Warning: --train_ground_truth_path not provided or file not found. Skipping evaluation on TRAIN set.")
        
        exit()

    # --- 正常预测+评估流程 ---
    if args.demand:
        data_type = f'recovered_by_{args.recovery_model_name}'
        target_col = 'sale_amount_pred'
        demand_df = pd.read_parquet(args.demand_path)
    else:
        data_type = 'censored'
        target_col = 'sale_amount'
        dataset = load_from_disk("../../frn_50k_local_dataset")
        demand_df = dataset['train'].to_pandas()

    # 1. 在训练集上预测和评估
    df_train_pred = predict_and_evaluate_on_train_split(demand_df.copy(), target_col, data_type)
    # --- Gemini Code Assist: Use the new --train_ground_truth_path argument ---
    train_truth_path = args.train_ground_truth_path if args.train_ground_truth_path else (args.demand_path if args.demand else None)
    if train_truth_path and os.path.exists(train_truth_path):
        evaluation_vs_recovered(df_train_pred, train_truth_path, 'sale_amount_pred', data_type, is_train_set=True)

    # 2. 在评估集上预测和评估
    pdf = seasonal_naive_predict(demand_df, target_col, recovery_model_name=args.recovery_model_name)
    if args.recovered_eval_path:
        evaluation_vs_recovered(pdf, args.recovered_eval_path, 'sale_amount_pred', data_type, is_train_set=False)
