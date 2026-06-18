import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import warnings

import lightgbm as lgb
import gc

warnings.filterwarnings("ignore")

def get_data(recovery_model_name, base_path):
    """Loads and prepares training and testing datasets."""
    print(f"Loading data for recovery model: {recovery_model_name}")
    train_path = base_path / f"latent_demand_recovery/exp/demand/demand_{recovery_model_name}.parquet"
    test_path = base_path / f"latent_demand_recovery/exp/demand/demand_eval_{recovery_model_name}.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Data files not found for model '{recovery_model_name}' at {train_path.parent}")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    required_col = [
        'city_id', 'store_id', 'first_category_id', 'second_category_id', 'product_id', 'dt', 'sale_amount_pred',
        'discount', 'holiday_flag', 'activity_flag', 'precpt', 'avg_temperature',
    ]
    if not all(col in df_train.columns for col in required_col):
        raise ValueError(f"Columns not found for train split")
    if not all(col in df_test.columns for col in required_col):
        raise ValueError(f"Columns not found for train split")

    df_full = pd.concat([df_train[required_col], df_test[required_col]], axis=0)
    df_full['unique_id'] = df_full['city_id'].astype(str) + '_' + df_full['product_id'].astype(str)
    df_full = df_full.rename(columns={'dt': 'ds', 'sale_amount_pred': 'y'})
    df_full['ds'] = pd.to_datetime(df_full['ds'])
   
    # Sort for time series operations
    df_full = df_full.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    print(f"full dataset: ")
    print(df_full.info())

    return df_full

def calculate_metrics(df, model_name):
    """Calculates WAPE, WPE, and ME, safely handling NaNs."""
    # Ensure no NaNs in prediction or true values for the comparison
    df_eval = df[['y', model_name]].dropna()
    y_true = df_eval['y']
    y_pred = df_eval[model_name]

    # Avoid division by zero
    y_true_sum = np.sum(y_true)
    if y_true_sum == 0:
        # If true sum is 0, WAPE is 0 if preds are also all 0, otherwise it's undefined/inf.
        # ME is the mean of the negative predictions.
        return {
            "WAPE": 0 if np.sum(np.abs(y_pred)) == 0 else np.inf,
            "WPE": np.sum(y_pred - y_true) / 1e-9,
            "ME": np.mean(y_pred - y_true)
        }

    wape = np.sum(np.abs(y_pred - y_true)) / y_true_sum
    wpe = np.sum(y_pred - y_true) / y_true_sum
    me = np.mean(y_pred - y_true)
    return {"WAPE": wape, "WPE": wpe, "ME": me}

def calculate_group_metrics(df, model_name, group_cols):
    """Calculate WAPE/WPE/ME for each group; returns DataFrame with group_type/group_value."""
    rows = []
    for g in group_cols:
        if g not in df.columns:
            continue
        for val, sub in df.groupby(g):
            sub_eval = sub[['y', model_name]].dropna()
            if sub_eval.empty:
                continue
            y_true = sub_eval['y']
            y_pred = sub_eval[model_name]
            y_true_sum = np.sum(y_true)
            if y_true_sum == 0:
                wape = 0 if np.sum(np.abs(y_pred)) == 0 else np.inf
                wpe = np.sum(y_pred  - y_true) / 1e-9 if y_true_sum == 0 else np.sum(y_pred  - y_true) / y_true_sum
            else:
                wape = np.sum(np.abs(y_pred - y_true)) / y_true_sum
                wpe = np.sum(y_pred  - y_true) / y_true_sum
            me = np.mean(y_pred  - y_true)
            rows.append({
                "group_type": g,
                "group_value": val,
                "count": len(sub_eval),
                "WAPE": wape,
                "WPE": wpe,
                "ME": me
            })
    return pd.DataFrame(rows)

# --- Model 1: Weekday (Naive) ---
def weekday_forecast(df_full, train_end_date, val_end_date):
    """
    Generates weekday-based (naive) forecasts.
    The forecast for day D is the value from day D-7.
    """
    print("\nRunning Weekday (Naive) forecast...")
    df_naive = df_full.copy()

    # The prediction is the value from 7 days ago, for a 7-day seasonality.
    df_naive['weekday'] = df_naive.groupby('unique_id')['y'].shift(7)

    # Filter for validation and test periods
    val_df = df_naive[(df_naive['ds'] <= train_end_date) & (df_naive['ds'] <= val_end_date)].copy()
    test_df = df_naive[(df_naive['ds'] > train_end_date) & (df_naive['ds'] <= val_end_date)].copy()

    return val_df, test_df

# --- Model 2: LightGBM ---
def create_lgb_features(df):
    """Create time series features based on time index."""
    df = df.copy()
    # Keep key static identifiers as categorical features so trees split on equality, not magnitude
    static_cats = ['city_id', 'store_id', 'product_id']
    for col in static_cats:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Day-level context features kept as numeric
    context_cols = ['discount', 'holiday_flag', 'activity_flag']
    for col in context_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df['dayofweek'] = df['ds'].dt.dayofweek
    df['weekofyear'] = df['ds'].dt.isocalendar().week
    df['month'] = df['ds'].dt.month
    df['dayofmonth'] = df['ds'].dt.day
    df['year'] = df['ds'].dt.year
    
    # Add lag features, starting from the forecast horizon
    print("Creating lag features...")
    lags = [7, 14]  # Using 7 and 14 day lags for weekly seasonality
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('unique_id')['y'].shift(lag)
        
    # Add rolling mean features
    print("Creating rolling window features...")
    rolling_windows = [7] # Reduced to a 7-day window
    for window in rolling_windows:
        # shift(7) because we are forecasting 7 days ahead
        df[f'rolling_mean_{window}'] = df.groupby('unique_id')['y'].shift(7).rolling(window).mean()
        df[f'rolling_std_{window}'] = df.groupby('unique_id')['y'].shift(7).rolling(window).std()

    df = df.dropna().reset_index(drop=True)
    gc.collect()
    return df

def train_and_predict_lgb(df_train, df_val, df_test):
    """Train a LightGBM model and make predictions."""
    
    features = [col for col in df_train.columns if col not in ['ds', 'y', 'unique_id']]
    # Ensure key static/context columns are present
    required_features = ['city_id', 'store_id', 'product_id', 'discount', 'holiday_flag', 'activity_flag']
    missing = [f for f in required_features if f not in features]
    if missing:
        raise ValueError(f"Missing required features for LightGBM: {missing}")

    # Treat ids as categorical to let trees split on equality
    categorical_features = ['city_id', 'store_id', 'product_id', 'dayofweek', 'weekofyear', 'month', 'year']

    print(f"\nTraining LightGBM with {len(features)} features...")
    
    # Create datasets
    X_train, y_train = df_train[features], df_train['y']
    X_val, y_val = df_val[features], df_val['y']
    X_test = df_test[features]

    lgb_train = lgb.Dataset(X_train, y_train, feature_name=features, 
                            categorical_feature=[f for f in categorical_features if f in features])
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(100, verbose=True)],
    )
    
    print("\nPredicting on validation and test sets...")
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    
    return val_preds, test_preds

def main():
    parser = argparse.ArgumentParser(description="Run time series forecasting.")
    parser.add_argument("--recovery_model_name", type=str, required=True, help="Name of the recovery model.")
    args = parser.parse_args()

    base_path = Path('.')
    df_full = get_data(args.recovery_model_name, base_path)

    # --- Run Models ---
    results = {}
    
    # --- Weekday (Naive) ---
    # We use the original dataframe for this simple model
    all_dates_orig = sorted(df_full['ds'].unique())
    train_end_date_orig = all_dates_orig[89]
    print(f"train_end_date_orig: {train_end_date_orig}")
    val_end_date_orig = all_dates_orig[96] if len(all_dates_orig) > 96 else all_dates_orig[-1]
    print(f"val_end_date_orig: {val_end_date_orig}")
    
    val_naive, test_naive = weekday_forecast(df_full, train_end_date_orig, val_end_date_orig)
    val_naive_metrics = calculate_metrics(val_naive, 'weekday')
    test_naive_metrics = calculate_metrics(test_naive, 'weekday')
    results['Weekday'] = {'Validation': val_naive_metrics, 'Test': test_naive_metrics}

    # --- LightGBM ---
    # Create features for the entire dataset first
    print("\nCreating features for LightGBM model...")
    df_featured = create_lgb_features(df_full)
    print(df_featured.head())
    print(df_featured.info())

    # To ensure the split is correct after dropna(), we find the split points
    # beforehand and select data based on the dates.
    train_end_date = all_dates_orig[89]
    val_end_date = all_dates_orig[96] if len(all_dates_orig) > 90 else train_end_date

    # Split the featured dataframe
    df_train = df_featured[df_featured['ds'] <= train_end_date]
    
    # We need a validation set for early stopping. Let's use the last 7 days of the training set.
    train_cutoff = train_end_date - pd.Timedelta(days=7)
    df_train_sub = df_train[df_train['ds'] <= train_cutoff]
    df_val = df_train[df_train['ds'] > train_cutoff]
    
    df_test = df_featured[df_featured['ds'] > train_end_date]
    print(df_train.head())
    print(df_train.info())
    print(df_val.head())
    print(df_val.info())

    if df_train_sub.empty or df_val.empty:
        raise ValueError("Training or validation set is empty after splitting. "
                         "This may be due to aggressive feature engineering and a short time series. "
                         "Try reducing the lag or rolling window sizes.")

    val_preds, test_preds = train_and_predict_lgb(df_train_sub, df_val, df_test)
    
    # Assign predictions for metrics calculation
    df_val_lgb = df_val.copy()
    df_val_lgb['LightGBM'] = val_preds
    val_lgb_metrics = calculate_metrics(df_val_lgb, 'LightGBM')
    
    df_test_lgb = df_test.copy()
    df_test_lgb['LightGBM'] = test_preds
    test_lgb_metrics = calculate_metrics(df_test_lgb, 'LightGBM')
    
    results['LightGBM'] = {'Validation': val_lgb_metrics, 'Test': test_lgb_metrics}

    # Grouped metrics for bias correction (store/city/category/weekday)
    # group_cols = ['store_id', 'city_id', 'first_category_id', 'second_category_id', 'dayofweek']
    # val_group_metrics = calculate_group_metrics(df_val_lgb, 'LightGBM', group_cols)
    # test_group_metrics = calculate_group_metrics(df_test_lgb, 'LightGBM', group_cols)
    
    # --- Print and Save Results ---
    print("\n--- Forecasting Results ---")
    for model, metrics in results.items():
        print(f"\nModel: {model}")
        print(f"  Validation Metrics: {metrics['Validation']}")
        print(f"  Test Metrics: {metrics['Test']}")
        
    results_path = base_path / "demand_forecasting" / "ClassicalModels" / "results"
    results_path.mkdir(exist_ok=True)

    # --- Save Metrics ---
    for model_name, metrics in results.items():
        # Validation metrics
        val_metrics_df = pd.DataFrame([metrics['Validation']])
        val_metrics_df.insert(0, 'subset', 'All Eval')
        val_metrics_filename = results_path / f"{model_name}_train_metrics_recovered_by_{args.recovery_model_name}.csv"
        val_metrics_df.to_csv(val_metrics_filename, index=False)

        # Test metrics
        test_metrics_df = pd.DataFrame([metrics['Test']])
        test_metrics_df.insert(0, 'subset', 'All Eval')
        test_metrics_filename = results_path / f"{model_name}_metrics_recovered_by_{args.recovery_model_name}.csv"
        test_metrics_df.to_csv(test_metrics_filename, index=False)

    # --- Save grouped metrics for LightGBM ---
    # if not val_group_metrics.empty:
    #     val_group_metrics.insert(0, 'subset', 'All Eval')
    #     val_group_metrics.to_csv(results_path / f"LightGBM_train_group_metrics_recovered_by_{args.recovery_model_name}.csv", index=False)
    # if not test_group_metrics.empty:
    #     test_group_metrics.insert(0, 'subset', 'All Eval')
    #     test_group_metrics.to_csv(results_path / f"LightGBM_group_metrics_recovered_by_{args.recovery_model_name}.csv", index=False)

    # --- Save Predictions for LightGBM ---
    id_cols = ['city_id', 'store_id', 'product_id']
    
    # Validation predictions
    df_val_pred_out = df_val_lgb[id_cols + ['ds']].copy()
    df_val_pred_out['sale_amount_pred'] = df_val_lgb['LightGBM']
    df_val_pred_out = df_val_pred_out.rename(columns={'ds': 'dt'})
    df_val_pred_out['dt'] = df_val_pred_out['dt'].dt.strftime('%Y-%m-%d')
    df_val_pred_out.to_parquet(results_path / f"LightGBM_train_predictions_recovered_by_{args.recovery_model_name}.parquet", index=False)

    # Test predictions
    df_test_pred_out = df_test_lgb[id_cols + ['ds']].copy()
    df_test_pred_out['sale_amount_pred'] = df_test_lgb['LightGBM']
    df_test_pred_out = df_test_pred_out.rename(columns={'ds': 'dt'})
    df_test_pred_out['dt'] = df_test_pred_out['dt'].dt.strftime('%Y-%m-%d')
    df_test_pred_out.to_parquet(results_path / f"LightGBM_predictions_recovered_by_{args.recovery_model_name}.parquet", index=False)

    # --- Save Predictions for Naive ---
    # Validation predictions
    df_val_naive_out = val_naive[id_cols + ['ds']].copy()
    df_val_naive_out['sale_amount_pred'] = val_naive['weekday']
    df_val_naive_out = df_val_naive_out.rename(columns={'ds': 'dt'})
    df_val_naive_out['dt'] = df_val_naive_out['dt'].dt.strftime('%Y-%m-%d')
    df_val_naive_out.to_parquet(results_path / f"Weekday_train_predictions_recovered_by_{args.recovery_model_name}.parquet", index=False)

    # Test predictions
    df_test_naive_out = test_naive[id_cols + ['ds']].copy()
    df_test_naive_out['sale_amount_pred'] = test_naive['weekday']
    df_test_naive_out = df_test_naive_out.rename(columns={'ds': 'dt'})
    df_test_naive_out['dt'] = df_test_naive_out['dt'].dt.strftime('%Y-%m-%d')
    df_test_naive_out.to_parquet(results_path / f"Weekday_predictions_recovered_by_{args.recovery_model_name}.parquet", index=False)

    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
