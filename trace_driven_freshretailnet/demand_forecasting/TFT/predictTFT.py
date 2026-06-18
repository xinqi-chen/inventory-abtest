import os
import configs.tft_config as config
import numpy as np
import pandas as pd
import torch
from trainer.model import Model
from dataset.dataset import Dataset
from models.tft.model import TemporalFusionTransformer
import time
from datetime import datetime, timedelta
from datasets import load_from_disk
import argparse

config.date = '2024-06-26'

config.use_gpu = True
config.num_workers = 0
config.dataset_config["max_prediction_length"] = 7
config.dataset_config["max_encoder_length"] = 70
target_dates = pd.date_range(start = config.date, periods = config.dataset_config["max_prediction_length"])
target_dates = [date.strftime("%Y-%m-%d") for date in target_dates]
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

import random
import pytorch_lightning as pl

fix_seed = 2025
os.environ["PYTHONHASHSEED"] = str(fix_seed)
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
pl.seed_everything(fix_seed, workers=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_prediction_file_paths(data_type):
    """Return default file paths for train/eval predictions based on data type."""
    if 'censored' in data_type:
        train_filename = "TFT_train_predictions_censored.parquet"
        eval_filename = "TFT_predictions_censored.parquet"
    else:
        train_filename = f"TFT_train_predictions_{data_type}.parquet"
        eval_filename = f"TFT_predictions_{data_type}.parquet"
    return (
        os.path.join(RESULTS_DIR, train_filename),
        os.path.join(RESULTS_DIR, eval_filename),
    )


def load_saved_predictions(path, split_name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {split_name} predictions at '{path}', but the file does not exist. "
            "Ensure Step 1 has finished generating predictions."
        )
    df = pd.read_parquet(path)
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'])
    return df

def loadDataset(args):
    """
    Loads train and eval splits based on command-line arguments.
    - If --demand is specified, loads recovered data.
    - Otherwise, loads original censored data from the local disk dataset.
    """
    t0 = time.time()
    if args.demand:
        print(f"--- Loading RECOVERED data from path: {args.demand_path} ---")
        df_train = pd.read_parquet(args.demand_path)
        df_train['sale_amount'] = df_train['sale_amount_pred']
        
        if not args.recovered_eval_path or not os.path.exists(args.recovered_eval_path):
            raise FileNotFoundError(f"Evaluation file for recovered demand not found at: {args.recovered_eval_path}")
        df_eval = pd.read_parquet(args.recovered_eval_path)
        # --- Gemini Code Assist: Ensure eval set also uses recovered demand as target ---
        df_eval['sale_amount'] = df_eval['sale_amount_pred']
        # -----------------------------------------------------------------------------
        # The psd (per-store-daily) should be calculated from the recovered training data
        df_train_psd = df_train.groupby(config.dataset_config['group_ids'])['sale_amount'].mean().to_frame('psd')
    else:
        print("--- Loading CENSORED data from local disk dataset ---")
        dataset = load_from_disk("../../frn_50k_local_dataset")
        df_train = dataset['train'].to_pandas()
        df_eval = dataset['eval'].to_pandas()
        # The psd is calculated from the original censored training data
        df_train_psd = df_train.groupby(config.dataset_config['group_ids'])['sale_amount'].mean().to_frame('psd')

    df_eval = pd.merge(df_eval, df_train_psd, on=config.dataset_config['group_ids'], how='left').fillna(0)
    return df_train, df_eval

def do_predict(dataset, model_path, prediction_type, target_dates, verbose=False):
    best_tft = TemporalFusionTransformer.load_from_checkpoint(model_path).cuda()

    t1 = time.time()
    # For deterministic, mode is 'prediction'. For quantile, it's 'quantiles'.
    predict_mode = "quantiles" if prediction_type == "quantile" else "prediction"
    predictions, index = best_tft.predict(dataset.predict_df, return_index=True, mode=predict_mode)
    t2 = time.time()
    print(f"predict cost {t2-t1}") if verbose else None

    # --- Gemini Code Assist: Handle both quantile and deterministic outputs ---
    if prediction_type == 'quantile':
        # Create column names for each quantile, e.g., 'q_0.05', 'q_0.50', etc.
        quantile_columns = [f"q_{q:.2f}" for q in config.quantiles]
        # Reshape predictions and create a DataFrame
        # Shape: (n_samples, n_timesteps, n_quantiles) -> (n_samples * n_timesteps, n_quantiles)
        preds_df = pd.DataFrame(predictions.reshape(-1, len(config.quantiles)), columns=quantile_columns)
        if 'q_0.50' in preds_df.columns:
            preds_df['sale_amount_pred'] = preds_df['q_0.50']
    else: # deterministic
        # Shape: (n_samples, n_timesteps) -> (n_samples * n_timesteps, 1)
        preds_df = pd.DataFrame(predictions.flatten(), columns=['sale_amount_pred'])
    # -------------------------------------------------------------------------

    preds_df[preds_df < 0] = 0 # Ensure non-negative predictions

    # Reconstruct the index (groups and dates) for the predictions
    idx = np.array(index[config.dataset_config['group_ids']])
    # Repeat each group ID for each of the 7 prediction days
    index_df = pd.DataFrame(np.repeat(idx, config.dataset_config["max_prediction_length"], axis=0), columns=config.dataset_config["group_ids"])
    # Create a date column that cycles through the target dates
    index_df['dt'] = np.tile(pd.to_datetime(target_dates), len(idx))
    index_df[['store_id', 'product_id']] = index_df[['store_id', 'product_id']].astype(int)

    # Combine index and predictions and return
    return pd.concat([index_df, preds_df], axis=1)

def cal_metrics(df, data_type, groups=["psd>=0"]):
    # This function is deprecated. All metric calculations are now handled by evaluation_vs_recovered.
    return pd.DataFrame()

def evaluation_vs_recovered(df_pred, ground_truth_path, data_type, prediction_type, is_train_set=False):
    """Compares TFT predictions against recovered demand from the eval set."""
    print("\n--- Evaluating Forecast vs. Recovered Demand (on Eval Set) ---")
    try:
        ground_truth_df = pd.read_parquet(ground_truth_path)
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {ground_truth_path}. Skipping this evaluation.")
        return

    # --- Gemini Code Assist: Determine metric filename based on user request ---
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    # Extract the recovery model name from the evaluation path
    # e.g., from '../../.../demand_eval_TimesNet.parquet' -> 'TimesNet'
    eval_model_name = os.path.basename(ground_truth_path).replace('demand_eval_', '').replace('demand_', '').replace('.parquet', '')

    train_or_eval_str = "train_" if is_train_set else ""

    if 'censored' in data_type:
        # Case 1: Input was 'raw', evaluating against a recovered demand.
        metrics_filename = f"TFT_{train_or_eval_str}metrics_censored_by_{eval_model_name}.csv"
    else:
        # Case 2: Input was a recovered model, evaluating against its own ground truth.
        # The eval_model_name should match the recovery model name from data_type.
        # For training set, we also evaluate against the same recovery model
        recovery_model_from_data_type = data_type.replace('recovered_by_', '')
        eval_model_name = recovery_model_from_data_type
        metrics_filename = f"TFT_{train_or_eval_str}metrics_recovered_by_{eval_model_name}.csv"

    # --- Gemini Code Assist: Fix merge error by ensuring consistent key types ---
    # Convert 'dt' columns in both dataframes to string format 'YYYY-MM-DD' before merging.
    # This prevents potential "You are trying to merge on datetime64[ns] and object columns" errors.
    df_pred['dt'] = pd.to_datetime(df_pred['dt']).dt.strftime('%Y-%m-%d')
    ground_truth_df['dt'] = pd.to_datetime(ground_truth_df['dt']).dt.strftime('%Y-%m-%d')

    # --- Gemini Code Assist: Rename ground truth column *before* merging to avoid name collision ---
    ground_truth_df = ground_truth_df.rename(columns={'sale_amount_pred': 'recovered_demand_truth'})

    # Merge forecast predictions with recovered demand ground truth
    merged_df = pd.merge(
        df_pred,
        ground_truth_df[['store_id', 'product_id', 'dt', 'recovered_demand_truth']],
        on=['store_id', 'product_id', 'dt']
    )

    # --- Gemini Code Assist: Select correct prediction column ---
    if prediction_type == 'quantile':
        pred_col = 'q_0.50'
    elif 'sale_amount_pred' in merged_df.columns:
        pred_col = 'sale_amount_pred'
    else:
        print(f"❌ Error: Cannot find a valid prediction column in the merged dataframe for metrics calculation.")
        return

    if 'stock_hour6_22_cnt' not in merged_df.columns:
        merged_df['stock_hour6_22_cnt'] = 0 # Assume no stockout if column is missing (e.g., on train set)

    all_metrics = []
    for subset_name, query in [
        ("All Eval", "stock_hour6_22_cnt >= 0"),
        ("Non-Stockout", "stock_hour6_22_cnt == 0"),
        ("Stockout", "stock_hour6_22_cnt > 0"),
    ]:
        subset_df = merged_df.query(query)
        if subset_df.empty: continue
        denominator = subset_df['recovered_demand_truth'].sum() + 1e-8
        wape = (subset_df[pred_col] - subset_df['recovered_demand_truth']).abs().sum() / denominator
        wpe = (subset_df[pred_col] - subset_df['recovered_demand_truth']).sum() / denominator
        me = (subset_df[pred_col] - subset_df['recovered_demand_truth']).mean()
        all_metrics.append({'subset': subset_name, 'wape': round(wape, 4), 'wpe': round(wpe, 4), 'me': round(me, 4)})
    
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df)
    metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
    print(f"TFT metrics saved to {os.path.join(results_dir, metrics_filename)}")

def get_sorted_files(directory='.'):
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    sorted_files = sorted(files)
    return sorted_files

# def get_pred(df_train, df_eval, data_type, prediction_type, start_epoch=0, verbose=False):
#     model_dir = f"./lightning_logs/{data_type}/checkpoints" # data_type already contains recovery model name
#     files = get_sorted_files(model_dir)
#     df_pred = []

#     # --- Gemini Code Assist: Create dataset object specifically for eval prediction ---
#     t0 = time.time()
#     df_train['dt'] = pd.to_datetime(df_train['dt'])
#     df_eval['dt'] = pd.to_datetime(df_eval['dt'])
#     df_full = pd.concat([df_train, df_eval], ignore_index=True)
#     df_full['day_of_week'] = df_full['dt'].dt.dayofweek
#     df_full.loc[df_full.dt >= config.date, 'sale_amount'] = np.nan
#     dataset_eval = Dataset(df_full, config)
#     t1 = time.time()
#     print(f"Eval dataset generation cost {t1-t0}")
#     # --------------------------------------------------------------------------------

#     for idx, file in enumerate(files):
#         epoch_num = file.split('-')[0].split('=')[1]
#         if start_epoch == -1:
#             if idx + 1 == len(files): # Only process the last file
#                 _data_type = f"{data_type}_epoch{epoch_num}"
#                 print(_data_type)
#                 print(f"Predicting with the last checkpoint: epoch {epoch_num}")
#                 prediction_df = do_predict(dataset_eval, os.path.join(model_dir, file), prediction_type, verbose)
#                 # For metrics, we use the full data_type which includes the recovery model name
#                 results_dir = RESULTS_DIR
#                 os.makedirs(results_dir, exist_ok=True)

#                 # --- Gemini Code Assist: Implement user's requested naming logic ---
#                 if 'censored' in data_type:
#                     prediction_filename = 'TFT_predictions_censored.parquet'
#                 else:
#                     prediction_filename = f'TFT_predictions_{data_type}.parquet'
                
#                 # The prediction_df from do_predict doesn't have ground truth yet.
#                 # We need to merge it first.
#                 df_groudtruth = df_eval[['store_id', 'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt', 'psd']]
#                 merged_df = pd.merge(prediction_df, df_groudtruth, on=['store_id', 'product_id', 'dt'], how='left')
#                 merged_df.to_parquet(os.path.join(results_dir, prediction_filename), index=False)
#                 print(f"TFT predictions saved to {os.path.join(results_dir, prediction_filename)}")
#                 df_pred.append((_data_type, merged_df))
#         else: # Legacy multi-epoch logic, can be removed later
#             if int(epoch_num) >= start_epoch:
#                 _data_type = f"{data_type}_epoch{epoch_num}"
#                 print(_data_type)
#                 print(f"Predicting with checkpoint: epoch {epoch_num}")
#                 prediction_df = do_predict(dataset_eval, os.path.join(model_dir, file), prediction_type, verbose)
#                 df_groudtruth = df_eval[['store_id', 'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt', 'psd']]
#                 merged_df = pd.merge(prediction_df, df_groudtruth, on=['store_id', 'product_id', 'dt'], how='left')
#                 df_pred.append((f"{data_type}_epoch{epoch_num}", merged_df))

#                 # --- Gemini Code Assist: Saving prediction results ---
#                 results_dir = RESULTS_DIR
#                 os.makedirs(results_dir, exist_ok=True)
#                 output_filename = f'TFT_predictions_{_data_type}.parquet' # _data_type is unique
#                 # For multi-epoch prediction, keep epoch in filename
#                 output_filename = f'TFT_predictions_{data_type}_epoch{epoch_num}.parquet'
#                 merged_df.to_parquet(os.path.join(results_dir, output_filename), index=False)
#                 print(f"TFT predictions for {_data_type} saved to {os.path.join(results_dir, output_filename)}")
#     return df_pred

def get_metrics(df_pred, groups=["psd>=0"]):
    df_res = pd.DataFrame()
    for (data_type, df) in df_pred:
        df_ = cal_metrics(df, data_type, groups)
        df_res = pd.concat([df_res, df_], axis=0, ignore_index=True)
    return df_res

# def predict_and_evaluate_on_train_split(df_train, model_path, args, data_type):
#     """
#     Generates predictions and metrics for the training set by predicting on the train_dataloader.
#     """
#     print("\n--- Starting Prediction and Evaluation on TRAINING Set ---")

#     # --- Gemini Code Assist: Create dataset object specifically for train prediction ---
#     t0 = time.time()
#     df_train['dt'] = pd.to_datetime(df_train['dt'])
#     df_train['day_of_week'] = df_train['dt'].dt.dayofweek
#     dataset_train = Dataset(df_train, config)
#     train_dataloader = dataset_train.train_dataloader
#     t1 = time.time()
#     print(f"Train dataset generation cost {t1-t0}")
#     # 1. Generate predictions on the training dataloader
#     best_tft = TemporalFusionTransformer.load_from_checkpoint(model_path).cuda()
#     predict_mode = "quantiles" if args.prediction_type == "quantile" else "prediction"
    
#     raw_preds, raw_index = best_tft.predict(train_dataloader, mode=predict_mode, return_index=True)

#     # 2. Process and align predictions with ground truth
#     # The output of predict on a dataloader is a single tensor and a single dataframe.
#     preds = raw_preds.cpu()
#     index = raw_index

#     if args.prediction_type == 'quantile':
#         quantile_columns = [f"q_{q:.2f}" for q in config.quantiles]
#         preds_df = pd.DataFrame(preds.reshape(-1, len(config.quantiles)), columns=quantile_columns)
#     else:
#         preds_df = pd.DataFrame(preds.flatten(), columns=['sale_amount_pred'])
    
#     # # Reconstruct the full index with dates
#     # index_df = pd.DataFrame(np.repeat(index[config.dataset_config['group_ids']].to_numpy(), config.dataset_config["max_prediction_length"], axis=0), columns=config.dataset_config["group_ids"])
    
#     # # Create date range for each prediction window
#     # time_idx_start = index['time_idx_first_prediction'].numpy()
#     # start_date_of_dataset = train_dataloader.dataset.data['dt'].min()
#     # dates = start_date_of_dataset + pd.to_timedelta(np.repeat(time_idx_start, 7) + np.tile(np.arange(7), len(time_idx_start)), unit='D')
#     # index_df['dt'] = dates

#     # Reconstruct the full index with dates
#     index_df = pd.DataFrame(
#         np.repeat(index[config.dataset_config['group_ids']].to_numpy(),
#                 config.dataset_config["max_prediction_length"], axis=0),
#         columns=config.dataset_config["group_ids"]
#     )

#     # Use the same target_dates logic as in do_predict()
#     index_df['dt'] = np.tile(pd.to_datetime(target_dates),
#                             len(index[config.dataset_config['group_ids']]))
#     index_df[['store_id', 'product_id']] = index_df[['store_id', 'product_id']].astype(int)

#     df_train_pred = pd.concat([index_df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
#     df_train_pred[df_train_pred.select_dtypes(include=np.number).columns] = df_train_pred.select_dtypes(include=np.number).clip(lower=0)

#     # 3. Save training set predictions
#     results_dir = './results'
#     train_pred_filename = f"TFT_train_predictions_{data_type.replace('_by', '')}.parquet"
#     df_train_pred.to_parquet(os.path.join(results_dir, train_pred_filename), index=False)
#     print(f"TFT training set predictions saved to {os.path.join(results_dir, train_pred_filename)}")

#     # 4. Evaluate training set predictions
#     # The ground truth for the training set is the demand file itself.
#     if args.demand:
#         ground_truth_path = args.demand_path
#     else: # censored
#         # For censored train, the ground truth is in a different file for evaluation.
#         ground_truth_path = f'../../latent_demand_recovery/exp/demand/demand_{args.recovery_model_name}.parquet'

#     if ground_truth_path and os.path.exists(ground_truth_path):
#         evaluation_vs_recovered(df_train_pred, ground_truth_path, data_type, args.prediction_type, is_train_set=True)
#     else:
#         print(f"Warning: Training ground truth file not found at '{ground_truth_path}'. Skipping train set evaluation.")

# def predict_and_evaluate_on_train_split(df_train, model_path, args, data_type):
#     """
#     Generates predictions and metrics for the training set by predicting on the train_dataloader.
#     """
#     print("\n--- Starting Prediction and Evaluation on TRAINING Set ---")

#     # 1) 生成训练 dataloader
#     t0 = time.time()
#     df_train = df_train.copy()
#     df_train['dt'] = pd.to_datetime(df_train['dt'])
#     df_train['day_of_week'] = df_train['dt'].dt.dayofweek
#     dataset_train = Dataset(df_train, config)
#     train_dataloader = dataset_train.train_dataloader
#     t1 = time.time()
#     print(f"Train dataset generation cost {t1 - t0:.2f}s")

#     # 2) 加载已训练好的模型并预测
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     best_tft = TemporalFusionTransformer.load_from_checkpoint(model_path).to(device)
#     best_tft.eval()

#     predict_mode = "quantiles" if args.prediction_type == "quantile" else "prediction"
#     with torch.no_grad():
#         raw_preds, raw_index = best_tft.predict(train_dataloader, mode=predict_mode, return_index=True)

#     preds = raw_preds.detach().cpu()
#     index = raw_index

#     # 3) 组织预测列
#     if args.prediction_type == 'quantile':
#         quantile_columns = [f"q_{q:.2f}" for q in config.quantiles]
#         preds_df = pd.DataFrame(preds.reshape(-1, len(config.quantiles)), columns=quantile_columns)
#     else:
#         preds_df = pd.DataFrame(preds.flatten(), columns=['sale_amount_pred'])

#     # 4) 重建索引并补日期（与 do_predict 保持一致：使用 target_dates）
#     # 这里假设你已有 target_dates（由 config.date 开始的 pred_len 天）
#     target_dates = pd.date_range(start=config.date, periods=config.dataset_config["max_prediction_length"])
#     target_dates = [d.strftime("%Y-%m-%d") for d in target_dates]

#     index_df = pd.DataFrame(
#         np.repeat(index[config.dataset_config['group_ids']].to_numpy(),
#                   config.dataset_config["max_prediction_length"], axis=0),
#         columns=config.dataset_config["group_ids"]
#     )
    
#     index_df['dt'] = np.tile(pd.to_datetime(target_dates), len(index[config.dataset_config['group_ids']]))
    
#     # 一些 checkpoint / merge 需要 int 类型
#     if set(['store_id', 'product_id']).issubset(index_df.columns):
#         index_df[['store_id', 'product_id']] = index_df[['store_id', 'product_id']].astype(int)

#     df_train_pred = pd.concat([index_df.reset_index(drop=True),
#                                preds_df.reset_index(drop=True)], axis=1)

#     # 5) 非负裁剪
#     num_cols = df_train_pred.select_dtypes(include=np.number).columns
#     df_train_pred[num_cols] = df_train_pred[num_cols].clip(lower=0)

#     # 6) 保存训练集预测
#     results_dir = './results'
#     os.makedirs(results_dir, exist_ok=True)
#     train_pred_filename = f"TFT_train_predictions_{data_type.replace('_by', '')}.parquet"
#     df_train_pred.to_parquet(os.path.join(results_dir, train_pred_filename), index=False)
#     print(f"TFT training set predictions saved to {os.path.join(results_dir, train_pred_filename)}")

#     # 7) 训练集评估：对齐 ground truth 并输出指标（与你的 evaluation_vs_recovered 一致）
#     if args.demand:
#         ground_truth_path = args.demand_path
#     else:
#         # censored
#         ground_truth_path = f'../../latent_demand_recovery/exp/demand/demand_{args.recovery_model_name}.parquet'

#     if ground_truth_path and os.path.exists(ground_truth_path):
#         evaluation_vs_recovered(df_train_pred, ground_truth_path, data_type, args.prediction_type, is_train_set=True)
#     else:
#         print(f"Warning: Training ground truth file not found at '{ground_truth_path}'. Skipping train set evaluation.")
def predict_and_evaluate_on_train_split(df_train, model_path, args, data_type):
    """
    使用训练好的 TFT 模型，在“训练集的最后 pred_len 天”上做预测并评估。
    关键点：
    1) 将训练集最后 pred_len 天的 sale_amount 置 NaN，构造预测窗口；
    2) 用该 DataFrame 构建 Dataset，再调用 do_predict() 产出预测；
    3) 与训练集对应日期的真值对齐，计算 WAPE/WPE/ME。
    """
    print("\n--- Starting Prediction and Evaluation on TRAINING Set ---")

    train_pred_path, _ = get_prediction_file_paths(data_type)

    if args.skip_predict:
        df_train_pred = load_saved_predictions(train_pred_path, "training")
        target_dates_train = sorted(df_train_pred['dt'].unique())
        print(f"[INFO] Loaded existing training predictions from {train_pred_path}")
    else:
        df_train = df_train.copy()
        df_train['dt'] = pd.to_datetime(df_train['dt'])
        df_train['day_of_week'] = df_train['dt'].dt.dayofweek
        pred_len = int(config.dataset_config["max_prediction_length"])

        train_min_dt = df_train['dt'].min()
        train_max_dt = df_train['dt'].max()
        tail_end_dt  = train_max_dt
        tail_start_dt = train_max_dt - pd.Timedelta(days=pred_len - 1)
        target_dates_train = pd.date_range(start=tail_start_dt, end=tail_end_dt, freq="D")
        target_dates_train = target_dates_train[target_dates_train >= train_min_dt]

        df_for_train_pred = df_train.copy()
        df_for_train_pred.loc[df_for_train_pred['dt'].isin(target_dates_train), 'sale_amount'] = np.nan

        t0 = time.time()
        dataset_for_train_pred = Dataset(df_for_train_pred, config)
        t1 = time.time()
        print(f"[INFO] Train-pred dataset build cost {t1 - t0:.2f}s")


        

        df_train_pred = do_predict(dataset_for_train_pred, model_path, args.prediction_type, target_dates_train, verbose=True)

        # --- make dates comparable (INSERT HERE) ---
        target_dates_train_str = pd.to_datetime(target_dates_train).strftime('%Y-%m-%d').tolist()
        df_train_pred['dt'] = pd.to_datetime(df_train_pred['dt']).dt.strftime('%Y-%m-%d')
        df_train['dt']      = pd.to_datetime(df_train['dt']).dt.strftime('%Y-%m-%d')

        df_train_pred = df_train_pred[df_train_pred['dt'].isin(target_dates_train_str)]

        truth_cols = ['store_id', 'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt', 'psd']
        truth_cols = [c for c in truth_cols if c in df_train.columns]
        df_train_truth_tail = df_train[df_train['dt'].isin(target_dates_train_str)][truth_cols].copy()
        df_train_truth_tail.rename(columns={'sale_amount': 'sale_amount_truth'}, inplace=True)

        merged_preview = pd.merge(
            df_train_pred[['store_id','product_id','dt','sale_amount_pred']],
            df_train_truth_tail,
            on=['store_id','product_id','dt'],
            how='inner'
        )
        # -----------------------------------------
         
        df_train_pred.to_parquet(train_pred_path, index=False)
        print(f"[INFO] TFT training set predictions saved to {train_pred_path}")
        print(f"[INFO] Train pred rows: {len(df_train_pred)}, merged rows with truth: {len(merged_preview)}")

    ground_truth_path = args.train_ground_truth_path

    if not args.skip_predict and args.train_ground_truth_path is None:
        print("[INFO] No train ground truth provided; prediction-only run.")

    if ground_truth_path and os.path.exists(ground_truth_path):
        evaluation_vs_recovered(
            df_train_pred,
            ground_truth_path,
            data_type,
            args.prediction_type,
            is_train_set=True
        )
    else:
        print(f"[WARN] Training ground truth not found: '{ground_truth_path}'. Skipping train set evaluation.")


def predict_and_evaluate_on_eval_split(df_train, df_eval, model_path, args, data_type):
    """
    Generates predictions and metrics for the evaluation set.
    """
    print("\n--- Starting Prediction and Evaluation on EVALUATION Set ---")

    _, eval_pred_path = get_prediction_file_paths(data_type)

    if args.skip_predict:
        merged_df = load_saved_predictions(eval_pred_path, "evaluation")
        print(f"[INFO] Loaded existing evaluation predictions from {eval_pred_path}")
    else:
        t0 = time.time()
        df_train['dt'] = pd.to_datetime(df_train['dt'])
        df_eval['dt'] = pd.to_datetime(df_eval['dt'])
        df_full = pd.concat([df_train, df_eval], ignore_index=True)
        df_full['day_of_week'] = df_full['dt'].dt.dayofweek
        df_full.loc[df_full.dt >= config.date, 'sale_amount'] = np.nan
        dataset_eval = Dataset(df_full, config)
        t1 = time.time()
        print(f"Eval dataset generation cost {t1 - t0:.2f}s")

        prediction_df = do_predict(dataset_eval, model_path, args.prediction_type, target_dates, verbose=True)

        df_groudtruth = df_eval[['store_id', 'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt', 'psd']]
        merged_df = pd.merge(prediction_df, df_groudtruth, on=['store_id', 'product_id', 'dt'], how='left')
        merged_df.to_parquet(eval_pred_path, index=False)
        print(f"TFT predictions saved to {eval_pred_path}")

    if args.recovered_eval_path:
        evaluation_vs_recovered(merged_df, args.recovered_eval_path, data_type, args.prediction_type, is_train_set=False)
    else:
        print("Warning: --recovered_eval_path not provided. Skipping evaluation on eval set.")


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
        default="none",
        help="Name of the recovery model used, for file naming."
    )
    parser.add_argument(
        "--train_ground_truth_path",
        type=str,
        default=None,
        help="Path to the recovered demand parquet for training-set evaluation."
    )
    parser.add_argument(
        "--skip_predict",
        action='store_true',
        help="Skip the prediction step and only run evaluations."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="quantile",
        choices=['quantile', 'deterministic'],
        help="Type of prediction to generate ('quantile' or 'deterministic')."
    )
    args = parser.parse_args()
    
    # --- Gemini Code Assist: Configure prediction type from args ---
    config.prediction_type = args.prediction_type
    if args.prediction_type == 'quantile':
        print("🚀 Running TFT in Quantile Prediction mode.")
        config.quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    else:
        print("🎯 Running TFT in Deterministic Prediction mode.")
        config.quantiles = []

    if args.demand:
        data_type = f'recovered_by_{args.recovery_model_name}'
    else:
        # Per user request, the input is just 'raw', but we keep a unique identifier for logging.
        data_type = f'censored_by_raw'

    if args.train_ground_truth_path is None and args.demand:
        args.train_ground_truth_path = args.demand_path

    # --- Gemini Code Assist: Refactored prediction and evaluation flow ---
    # 1. Load data splits based on args
    df_train, df_eval = loadDataset(args)

    # 2. Get path to the trained model
    model_dir = f"./lightning_logs/{data_type}/checkpoints"
    last_checkpoint_file = get_sorted_files(model_dir)[-1]
    model_path = os.path.join(model_dir, last_checkpoint_file)

    # 3. Run prediction and evaluation for the TRAINING set
    predict_and_evaluate_on_train_split(df_train, model_path, args, data_type)

    # 4. Run prediction and evaluation for the EVALUATION set
    predict_and_evaluate_on_eval_split(df_train, df_eval, model_path, args, data_type)
