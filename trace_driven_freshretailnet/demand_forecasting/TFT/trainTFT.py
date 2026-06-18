import configs.tft_config as config
import os
import time, random
from datetime import datetime
import pandas as pd
import numpy as np
from trainer.model import Model
from dataset.dataset import Dataset
import torch
torch.set_float32_matmul_precision('high') # medium
from datasets import load_from_disk
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

import pytorch_lightning as pl

os.environ["PYTHONHASHSEED"] = str(fix_seed)
torch.cuda.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
pl.seed_everything(fix_seed, workers=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


root_path = os.path.dirname(os.path.abspath("./") + "/")
print(root_path)

# For multiple training, a new checkpoint callback must be set each time
def get_checkpoint_callback(data_type='censored', recovery_model_name='none'):
    return ModelCheckpoint(
        dirpath=f"./lightning_logs/{data_type}_by_{recovery_model_name}/checkpoints",
        filename="{epoch}-{step}",  # Filename format
        save_top_k=-1,  # Save all checkpoints (do not delete old files)
        every_n_epochs=1,  # Save once every epoch (optional, default is 1)
)

def run(df, config):
    t = time.time()
    dataset = Dataset(df, config)
    t0 = time.time()
    print(f"dataset generation cost {t0-t}")
    model = Model(dataset, config)
    t1 = time.time()
    print(f"model init cost {t1-t0}")
    model.train()
    t2 = time.time()
    print(f"train cost {t2-t1}")
    if config.valid:
        model.valid()
        t3 = time.time()
        print(f"validation cost {t3-t2}")

config.date = '2024-06-26'

config.batch_size = 1024
config.valid = False
config.use_gpu = True
config.num_workers = 4

config.dataset_config["min_prediction_length"] = 7
config.dataset_config["max_prediction_length"] = 7
config.dataset_config["min_encoder_length"] = 70
config.dataset_config["max_encoder_length"] = 70

config.model_config["learning_rate"] = 0.01
config.model_config["hidden_size"] =  128
config.model_config["attention_head_size"] = 8
config.model_config["dropout"] = 0.1

config.model_config['lstm_layers'] = 2
config.model_config['optimizer'] = 'adamw'  # ranger
config.model_config['weight_decay'] = 1e-4

config.trainer_config["max_epochs"] = 5
config.trainer_config["default_root_dir"] = root_path


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
        "--recovery_model_name",
        type=str,
        default="none",
        help="Name of the recovery model used, for file naming."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="quantile",
        choices=['quantile', 'deterministic'],
        help="Type of prediction to generate ('quantile' or 'deterministic')."
    )
    t1 = time.time()
    args = parser.parse_args()

    # --- Gemini Code Assist: Configure prediction type from args, removing global default ---
    config.prediction_type = args.prediction_type
    if args.prediction_type == 'quantile':
        print("🚀 Running TFT in Quantile Prediction mode.")
        config.quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    else:
        print("🎯 Running TFT in Deterministic Prediction mode.")
        config.quantiles = [] # Empty list for deterministic point forecasts (defaults to MSE loss)

    if args.demand:
        data_type = 'recovered'
        config.trainer_config['callbacks'] = get_checkpoint_callback(data_type, args.recovery_model_name)
        df = pd.read_parquet(args.demand_path)
        df['sale_amount'] = df['sale_amount_pred']
    else:
        config.trainer_config['callbacks'] = get_checkpoint_callback('censored', args.recovery_model_name)
        # --- Gemini Code Assist: Load dataset from local disk ---
        dataset = load_from_disk("../../frn_50k_local_dataset")
        # ------------------------------------------------------
        df = dataset['train'].to_pandas()
    df = df.sort_values(['city_id', 'store_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id', 'product_id', 'dt'])
    # --- Gemini Code Assist: Use vectorized datetime operations ---
    df['dt'] = pd.to_datetime(df['dt'])
    df['day_of_week'] = df['dt'].dt.dayofweek
    print(len(df))
    t2 = time.time()
    print(f"load data cost {t2 - t1}s")

    run(df, config)