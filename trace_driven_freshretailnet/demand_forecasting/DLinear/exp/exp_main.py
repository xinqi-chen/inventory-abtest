from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import dlinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_from_disk
from torch import optim
from tqdm import tqdm
import os
import time
import warnings
from lib.revin import RevIN
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # norm
        self.revin = RevIN(self.args.enc_in, eps=1e-5, affine=False)
        model = dlinear.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterition = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        writer = SummaryWriter(log_dir='./runs/' + setting)

        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in tqdm(range(self.args.train_epochs), desc="Training", unit="epoch"):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(f"\nEpoch {epoch + 1}, Start:")
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                else:
                    outputs = self.model(batch_x)
                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                    # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # writer.add_scalar('loss/train/batch', loss, iter_count + epoch * len(train_loader))

                if (i + 1) % 200 == 0:
                    dis_str = "\titers: {0:>5d}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    dis_str += "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time)
                    print(dis_str)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {}, cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            writer.add_scalar('loss/train/epoch', train_loss, epoch)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                writer.add_scalar('loss/vali/epoch', vali_loss, epoch)
                writer.add_scalar('loss/test/epoch', test_loss, epoch)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        writer.close()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # --- Gemini Code Assist: Calculate and save Mean Error on the training set ---
        print("\n--- Calculating Mean Error on Training Set ---")
        self.evaluate_train_set_me(train_data, train_loader)
        # --------------------------------------------------------------------------
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(
                './checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate(
                        (input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate(
                        (input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        # --- Gemini Code Assist: Implement --skip_predict logic ---
        # If skip_predict is True, we only perform evaluation against a recovered demand set.
        if self.args.skip_predict:
            if not self.args.recovered_eval_path:
                print("Error: --skip_predict requires --recovered_eval_path to be set.")
                return

            print("\n--- Running in Evaluation-Only Mode (skip_predict=True) ---")
            # Determine the path to the existing prediction file
            results_dir = './results/'
            if 'censored_raw' in self.args.des:
                prediction_filename = 'DLinear_predictions_censored.parquet'
            else:
                recovery_model = self.args.des.replace('recovered_', '')
                prediction_filename = f'DLinear_predictions_recovered_by_{recovery_model}.parquet'
            
            pred_file_path = os.path.join(results_dir, prediction_filename)
            print(f"Loading existing predictions from: {pred_file_path}")
            df_pred = pd.read_parquet(pred_file_path)

            self.evaluation_vs_recovered(df_pred, self.args.recovered_eval_path)
            return
        # --- End of skip_predict logic ---

        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(pred_loader, desc="Predicting", leave=True)):
                batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float()
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    # batch_size, pred_len, f_dim
                    outputs = self.model(batch_x)
                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                # f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                # b, pred_len, f_dim
                outputs = outputs[:, -self.args.pred_len:, :]
                # print("outputs", outputs.shape)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        # batch_size*, pred_len, f_dim
        preds = np.concatenate(preds, axis=0)
        # batch_size*pred_len, f_dim
        preds = preds.reshape(preds.shape[0] * preds.shape[1], preds.shape[2])
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        f_dim = -1 if self.args.features == 'MS' else 0
        preds = np.squeeze(preds[:, f_dim:])

        # --- Gemini Code Assist: Standardize output file naming ---
        # This change aligns DLinear's output with SSA and TFT for easier processing.

        results_dir = './results/'
        os.makedirs(results_dir, exist_ok=True)

        # Determine the run type (censored or recovered) from the 'des' argument
        if 'censored_raw' in self.args.des:
            # Per user request, simplify the name for raw input predictions
            predictions_filename = 'DLinear_predictions_censored.parquet'
        elif 'recovered_' in self.args.des:
            # Extract the recovery model name, e.g., from 'recovered_DLinear'
            recovery_model = self.args.des.replace('recovered_', '')
            predictions_filename = f'DLinear_predictions_recovered_by_{recovery_model}.parquet'
        else:
            # Fallback for any other case
            predictions_filename = f'DLinear_predictions_{self.args.des}.parquet'

        # The old daily metrics are no longer needed per user request.
        # metrics_filename = f'DLinear_metrics_{run_type_suffix}.parquet'
        # --------------------------------------------------------------------

        date = "2024-06-26"
        # pred数据集构造
        target_dates = pd.date_range(
            start=date,
            periods=self.args.pred_len,
        )
        target_dates = [date.strftime("%Y-%m-%d") for date in target_dates]
        # --- Gemini Code Assist: Pass numpy array directly to avoid temporary file ---
        # The data_organization function is updated to accept the numpy array directly.
        merged_df = self.data_organization(preds,
                                           target_dates=target_dates)
        # --- Gemini Code Assist: Saving prediction results ---
        merged_df.to_parquet(os.path.join(results_dir, predictions_filename))
        print(f"DLinear predictions saved to {os.path.join(results_dir, predictions_filename)}")
        
        # --- Gemini Code Assist: Add evaluation against recovered eval demand ---
        if self.args.recovered_eval_path:
            self.evaluation_vs_recovered(merged_df, self.args.recovered_eval_path)

        # --- Gemini Code Assist: Estimate residual distribution per PSD group ---
        print("\n--- Estimating Residual Distribution ---")
        # Calculate residuals on non-stockout days
        non_stockout_df = merged_df.query('stock_hour6_22_cnt == 0').copy()
        non_stockout_df['residual'] = non_stockout_df['sale_amount'] - non_stockout_df['sale_amount_pred']

        # Calculate standard deviation of residuals for each PSD group
        residual_std = non_stockout_df.groupby(non_stockout_df['psd'] >= 1)['residual'].std().to_dict()
        sigma_low_psd = residual_std.get(False, 0)  # PSD < 1
        sigma_high_psd = residual_std.get(True, 0)   # PSD >= 1

        print(f"Std Dev of Residuals (PSD < 1): {sigma_low_psd:.4f}")
        print(f"Std Dev of Residuals (PSD >= 1): {sigma_high_psd:.4f}")

        # Add the estimated sigma to each row in the prediction dataframe
        merged_df['pred_sigma'] = np.where(merged_df['psd'] < 1, sigma_low_psd, sigma_high_psd)
        
        # Re-save the prediction file with the new 'pred_sigma' column
        merged_df.to_parquet(os.path.join(results_dir, predictions_filename))
        print(f"DLinear predictions with 'pred_sigma' saved to {os.path.join(results_dir, predictions_filename)}")
        # --------------------------------------------------------------------

        return

    def data_organization(self, np_pred, target_dates):
        group_ids = ['store_id', 'product_id']
        # load data
        # --- Gemini Code Assist: Load dataset from local disk ---
        dataset = load_from_disk("../../frn_50k_local_dataset")
        # ------------------------------------------------------
        df_train = dataset['train'].to_pandas()
        df_eval = dataset['eval'].to_pandas()
        df_train_psd = df_train.groupby(
            group_ids)['sale_amount'].mean().to_frame('psd')
        df_eval = pd.merge(df_eval, df_train_psd, on=group_ids)

        # col rename
        df_train = df_train.rename(columns={'dt': 'date'})
        df_eval = df_eval.rename(columns={'dt': 'date'})
        # sort（必须）
        df_train = df_train.sort_values(by=['store_id', 'product_id', 'date'])
        df_eval = df_eval.sort_values(by=['store_id', 'product_id', 'date'])

        unique_groups = df_train[['store_id', 'product_id']
                                 ].drop_duplicates().reset_index(drop=True)
        # 创建日期数据框
        date_df = pd.DataFrame({'date': target_dates})
        # 使用 merge 进行笛卡尔积
        expanded_df = unique_groups.assign(key=1).merge(
            date_df.assign(key=1), on='key').drop('key', axis=1)
        np_pred[np_pred < 0] = 0 # noqa
        expanded_df['sale_amount_pred'] = np_pred
        df_pred = expanded_df

        # --- Gemini Code Assist: Ensure consistent datetime type before merge ---
        df_groudtruth = df_eval[['store_id', 'product_id', 'date', 'sale_amount', 'stock_hour6_22_cnt', 'psd']]
        # df_pred['date'] is already a string from target_dates, convert both to string for robust merge
        df_groudtruth['date'] = df_groudtruth['date'].astype(str)
        df_pred['date'] = df_pred['date'].astype(str)
        merged_df = pd.merge(df_groudtruth, df_pred, on=['store_id', 'product_id', 'date'])
        return merged_df

    def _calculate_and_save_metrics(self, merged_df, data_split, eval_model_name):
        """
        A unified function to calculate and save WAPE, WPE, and ME metrics.
        """
        results_dir = './results/'
        os.makedirs(results_dir, exist_ok=True)

        train_str = "train_" if data_split == 'train' else ""

        if 'censored_raw' in self.args.des:
            # For censored data, the eval model name comes from the ground truth file path
            metrics_filename = f"DLinear_{train_str}metrics_censored_by_{eval_model_name}.csv"
        else:
            recovery_model = self.args.des.replace('recovered_', '')
            metrics_filename = f"DLinear_{train_str}metrics_recovered_by_{recovery_model}.csv"

        all_metrics = []
        # Ensure stock_hour6_22_cnt exists, default to 0 if not (for train set)
        if 'stock_hour6_22_cnt' not in merged_df.columns:
            merged_df['stock_hour6_22_cnt'] = 0

        for subset_name, query in [
            ("All Eval", "stock_hour6_22_cnt >= 0"),
            ("Non-Stockout", "stock_hour6_22_cnt == 0"),
            ("Stockout", "stock_hour6_22_cnt > 0"),
        ]:
            subset_df = merged_df.query(query)
            if subset_df.empty:
                continue
            
            # Use 'recovered_demand_truth' as the ground truth column
            denominator = subset_df['recovered_demand_truth'].sum() + 1e-8
            
            wape = (subset_df['sale_amount_pred'] - subset_df['recovered_demand_truth']).abs().sum() / denominator
            wpe = (subset_df['sale_amount_pred'] - subset_df['recovered_demand_truth']).sum() / denominator
            me = (subset_df['sale_amount_pred'] - subset_df['recovered_demand_truth']).mean()
            
            all_metrics.append({
                'subset': subset_name,
                'wape': round(wape, 4),
                'wpe': round(wpe, 4),
                'me': round(me, 4)
            })

        if not all_metrics:
            print(f"No data to calculate metrics for {data_split} split.")
            return

        metrics_df = pd.DataFrame(all_metrics)
        print(f"\n--- Metrics for {data_split.upper()} Split ---")
        print(metrics_df)
        metrics_df.to_csv(os.path.join(results_dir, metrics_filename), index=False)
        print(f"DLinear {data_split} metrics saved to {os.path.join(results_dir, metrics_filename)}")

    # def evaluate_train_set_me(self, train_data, train_loader):
    #     """
    #     Calculates WAPE, WPE, and ME on the training set and saves them.
    #     This is called after training is complete and the best model is loaded.
    #     It re-runs prediction on a non-shuffled train loader to align predictions with ground truth attributes.
    #     """
    #     # 1. Create a new, non-shuffled dataloader for the training set
    #     train_data_for_pred, train_loader_for_pred = data_provider(self.args, flag='train', shuffle=False)

    #     self.model.eval()
    #     preds = []
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y) in enumerate(tqdm(train_loader_for_pred, desc="Evaluating on Train Set")):
    #             batch_x = batch_x.float().to(self.device)
    #             if self.args.revin:
    #                 batch_x = self.revin(batch_x, mode='norm')

    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     outputs = self.model(batch_x)
    #             else:
    #                 outputs = self.model(batch_x)
                
    #             if self.args.revin:
    #                 outputs = self.revin(outputs, mode='denorm')

    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #             preds.append(outputs.detach().cpu().numpy())

    #     preds = np.concatenate(preds, axis=0)
        
    #     # Reshape and inverse transform
    #     preds = preds.reshape(-1, preds.shape[-2] * preds.shape[-1])
    #     if train_data_for_pred.scale:
    #         preds = train_data_for_pred.inverse_transform(preds)

    #     # 2. Load the appropriate ground truth for the training set
    #     if self.args.data == 'recovered_dataset':
    #         # For recovered data, the ground truth is the training data file itself
    #         ground_truth_path = self.args.data_path
    #         eval_model_name = self.args.des.replace('recovered_', '')
    #     else:
    #         # For censored data, infer the train ground truth path from the eval path
    #         if not self.args.recovered_eval_path:
    #             print("Warning: Cannot determine ground truth for censored train set evaluation without --recovered_eval_path. Skipping.")
    #             return
    #         eval_model_name = os.path.basename(self.args.recovered_eval_path).replace('demand_eval_', '').replace('.parquet', '')
    #         ground_truth_path = os.path.join(os.path.dirname(self.args.recovered_eval_path), f'demand_{eval_model_name}.parquet')

    #     try:
    #         df_truth = pd.read_parquet(ground_truth_path)
    #     except FileNotFoundError:
    #         print(f"Warning: Train ground truth file not found at '{ground_truth_path}'. Skipping train set evaluation.")
    #         return

    #     # 3. Construct the prediction DataFrame
    #     # The predictions correspond to the last `pred_len` days of each sequence in the non-shuffled loader.
    #     # The `data_provider` creates sequences that slide over the data.
    #     # The number of predictable samples is `len(df_truth) - self.args.total_seq_len + 1`.
    #     # This is complex. A simpler way is to use the `data_organization` logic.
    #     # Let's reuse the logic from `predict()` to create a clean prediction dataframe.
        
    #     # We need to get the dates for the predictions.
    #     # The predictions are for the *end* of each sliding window in the training set.
    #     # Let's get the last date of each sequence from the data_provider.
        
    #     # This is getting overly complex. Let's simplify the goal:
    #     # We have `preds` (unshuffled predictions) and `df_truth` (unshuffled ground truth).
    #     # We need to align them. The `data_provider` creates `len(df) - total_len + 1` samples.
    #     # Each prediction corresponds to the 7 days following a 62-day history.
        
    #     # Let's use a more robust approach: create a prediction dataframe and merge.
    #     # The `data_provider` creates samples for each item. Let's get the unique items.
    #     group_ids = ['store_id', 'product_id']
    #     unique_groups = df_truth[group_ids].drop_duplicates()
        
    #     # The number of predictable windows per item is `90 - (62+7) + 1 = 22`.
    #     # This is still too complex. Let's use the simplest possible alignment.
    #     # The `train_data_for_pred.data_x` has the input sequences.
    #     # The last date of the first input sequence is the start of the first prediction.
        
    #     # Let's abandon this complex alignment and use a simpler, more direct approach.
    #     # We will re-use the `predict` function's logic on the training data.
        
    #     # Create a temporary 'pred' dataset from the training data
    #     self.args.data_flag = 'train' # Temporarily set flag for data_provider
    #     pred_data_train, _ = self._get_data(flag='pred')
    #     self.args.data_flag = 'test' # Reset flag
        
    #     # Re-run prediction on this 'pred' dataset
    #     # This is inefficient but guarantees correct alignment.
    #     # Let's stick to the previous implementation and accept the limitation for now.
    #     # The user's primary goal is the ME, which doesn't require stockout info.
        
    #     # Re-implementing the simplified version that was correct for the main goal.
    #     df_truth = pd.read_parquet(ground_truth_path)
    #     df_truth = df_truth.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
        
    #     # The `train_loader_for_pred` gives us `len(df_truth) - total_seq_len + 1` samples.
    #     # Let's get the true values that correspond to these predictions.
    #     true_values_for_pred = []
    #     for i in range(len(df_truth) - self.args.total_seq_len + 1):
    #         start = i + self.args.seq_len
    #         end = start + self.args.pred_len
    #         true_values_for_pred.append(df_truth['sale_amount_pred'].iloc[start:end].values)
        
    #     trues_unscaled = np.array(true_values_for_pred)
    #     preds_unscaled = preds.squeeze() # Predictions are already unscaled if they came from the same loader

    #     # Now we have aligned `preds_unscaled` and `trues_unscaled`.
    #     # But we still don't have stockout info aligned.
        
    #     # Final attempt at a robust solution:
    #     # Let's create the prediction dataframe with dates and merge.
    #     df_pred_train = self.data_organization(preds, df_truth)
    #     df_pred_train = df_pred_train.rename(columns={'sale_amount_pred': 'sale_amount_pred_model'})
        
    #     merged_df = pd.merge(df_pred_train, df_truth, on=['store_id', 'product_id', 'date'], how='inner')
    #     merged_df = merged_df.rename(columns={'sale_amount_pred': 'recovered_demand_truth', 'sale_amount_pred_model': 'sale_amount_pred'})

    #     self._calculate_and_save_metrics(merged_df, 'train', eval_model_name)

    def evaluate_train_set_me(self, train_data, train_loader):
        """
        在训练完成后，对训练集进行一次完整的前向预测。
        - 如果输入是 'recovered' 数据，则直接与自身的真值比较并计算指标。
        - 如果输入是 'censored' (raw) 数据，则遍历所有恢复模型，
          分别加载它们对应的训练集真值 (demand_{model}.parquet)，
          并计算和保存多份指标文件。
        """
        print("\n--- Evaluating Model on TRAINING Set ---")

        # 1) 用已有 train_data 构造“不打乱”的 DataLoader（不要再调用 data_provider）
        batch_size  = self.args.batch_size
        num_workers = self.args.num_workers
        drop_last   = bool(getattr(self.args, "drop_last", False))

        train_loader_for_pred = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,      # 评估时固定样本顺序
            num_workers=num_workers,
            drop_last=drop_last,
        )

        self.model.eval()

        preds_buf = []

        use_amp   = self.args.use_amp
        features  = self.args.features
        pred_len  = self.args.pred_len
        f_dim     = -1 if features == 'MS' else 0

        with torch.no_grad():
            for batch in train_loader_for_pred:
                # 兼容 (batch_x, batch_y, ...) / dict 两种返回
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0].float().to(self.device)
                    batch_y = batch[1].float()
                else:
                    batch_x = batch["batch_x"].float().to(self.device)
                    batch_y = batch["batch_y"].float()

                # 可选：RevIN 归一化
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')

                # 前向（可选 AMP）
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                # 可选：RevIN 反归一化（对输出）
                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                # 与训练/验证一致的切片：取最后 pred_len 步，形状 [B, L, C_pred]
                outputs = outputs[:, -pred_len:, f_dim:]

                preds_buf.append(outputs.detach().cpu())

        preds = torch.cat(preds_buf, dim=0).numpy()  # [N, L, C_pred]

        # 2) 反变换预测值
        if train_data.scale:
            N, L, C_pred = preds.shape
            num_features_scaler_fit_on = self.args.enc_in
            target_feature_idx = 0 # 假设目标列总是第一个

            temp_preds_full = np.zeros((N * L, num_features_scaler_fit_on), dtype=preds.dtype)
            temp_preds_full[:, target_feature_idx] = preds.reshape(-1)
            
            preds_unscaled_full = train_data.inverse_transform(temp_preds_full)
            preds_flat = preds_unscaled_full[:, target_feature_idx]
        else:
            preds_flat = preds.reshape(-1)

        # 重建预测的 (store_id, product_id, date) 信息，并仅保留每个序列最后 7 天
        df_source_sorted = train_data.data_pd.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)
        total_seq_len = self.args.total_seq_len
        pred_len = self.args.pred_len
        seq_len = self.args.seq_len

        num_samples = len(train_data)
        num_total_preds = num_samples * pred_len
        if len(preds_flat) != num_total_preds:
            print(f"[WARN] Prediction count mismatch. Expected {num_total_preds}, got {len(preds_flat)}. Truncating predictions.")
            preds_flat = preds_flat[:num_total_preds]

        pred_rows = []
        for _, group_df in df_source_sorted.groupby(['store_id', 'product_id']):
            for i in range(len(group_df) - total_seq_len + 1):
                pred_slice = group_df.iloc[i + seq_len: i + total_seq_len]
                pred_rows.extend(pred_slice[['store_id', 'product_id', 'date']].to_dict('records'))

        df_pred_with_ids = pd.DataFrame(pred_rows)
        df_pred_with_ids['date'] = pd.to_datetime(df_pred_with_ids['date'])
        df_pred_with_ids['sale_amount_pred'] = preds_flat[:len(df_pred_with_ids)]
        df_pred_with_ids = df_pred_with_ids.sort_values(['store_id', 'product_id', 'date'])
        df_pred_with_ids = df_pred_with_ids.drop_duplicates(subset=['store_id', 'product_id', 'date'], keep='last')

        # 保存train上的预测结果来看看
        if 'censored_raw' in self.args.des:
            # Per user request, simplify the name for raw input predictions
            predictions_filename = 'DLinear_train_predictions_censored.parquet'
        elif 'recovered_' in self.args.des:
            # Extract the recovery model name, e.g., from 'recovered_DLinear'
            recovery_model = self.args.des.replace('recovered_', '')
            predictions_filename = f'DLinear_train_predictions_recovered_by_{recovery_model}.parquet'
        else:
            # Fallback for any other case
            predictions_filename = f'DLinear_predictions_{self.args.des}.parquet'

        df_pred_last7 = (
            df_pred_with_ids
            .groupby(['store_id', 'product_id'], group_keys=False)
            .apply(lambda df: df.sort_values('date').tail(7))
            .reset_index(drop=True)
        )
        df_pred_last7['date'] = df_pred_last7['date'].dt.strftime('%Y-%m-%d')

        df_pred_with_ids = (
            df_pred_with_ids
            .groupby(['store_id', 'product_id'], group_keys=False)
            .apply(lambda df: df.sort_values('date'))
            .reset_index(drop=True)
        )
        df_pred_with_ids['date'] = df_pred_with_ids['date'].dt.strftime('%Y-%m-%d')
        df_pred_with_ids.to_parquet(os.path.join("./results/", predictions_filename))

        # 3) 根据输入类型（censored vs recovered）决定评估逻辑
        if 'censored_raw' in self.args.des:
            print("\n[INFO] Input is 'censored'. Evaluating against all recovery models' train sets (last 7 days).")
            all_recovery_models = ["TimesNet", "ImputeFormer", "SAITS", "iTransformer", "CSDI"]
            for eval_model_name in all_recovery_models:
                print(f"\n  -> Evaluating against TRAIN ground truth from: {eval_model_name}")
                truth_path = f'../../latent_demand_recovery/exp/demand/demand_{eval_model_name}.parquet'
                try:
                    df_truth = pd.read_parquet(truth_path)
                    df_truth = df_truth.rename(columns={'sale_amount_pred': 'recovered_demand_truth'})
                    df_truth['date'] = pd.to_datetime(df_truth['dt']).dt.strftime('%Y-%m-%d')
                    # df_truth_last7 = (
                    #     df_truth.sort_values(['store_id', 'product_id', 'date'])
                    #             .groupby(['store_id', 'product_id'], group_keys=False)
                    #             .apply(lambda df: df.tail(7))
                    #             .reset_index(drop=True)
                    # )

                    # truth_cols = ['store_id', 'product_id', 'date', 'recovered_demand_truth']
                    # if 'stock_hour6_22_cnt' in df_truth_last7.columns:
                    #     truth_cols.append('stock_hour6_22_cnt')

                    # merged_df = pd.merge(
                    #     df_pred_last7,
                    #     df_truth_last7[truth_cols],
                    #     on=['store_id', 'product_id', 'date'],
                    #     how='inner'
                    # )
                    df_truth = df_truth.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)

                    truth_cols = ['store_id', 'product_id', 'date', 'recovered_demand_truth']
                    if 'stock_hour6_22_cnt' in df_truth.columns:
                        truth_cols.append('stock_hour6_22_cnt')

                    merged_df = pd.merge(
                        df_pred_with_ids,
                        df_truth[truth_cols],
                        on=['store_id', 'product_id', 'date'],
                        how='inner'
                    )

                    self._calculate_and_save_metrics(
                        merged_df=merged_df,
                        data_split="train",
                        eval_model_name=eval_model_name
                    )
                except FileNotFoundError:
                    print(f"  [WARN] Ground truth file not found, skipping: {truth_path}")
        else:
            print("\n[INFO] Input is 'recovered'. Evaluating against its own train set (full series).")
            df_truth = pd.read_parquet(self.args.data_path)
            df_truth = df_truth.rename(columns={'sale_amount_pred': 'recovered_demand_truth'})
            df_truth['date'] = pd.to_datetime(df_truth['dt']).dt.strftime('%Y-%m-%d')

            truth_cols = ['store_id', 'product_id', 'date', 'recovered_demand_truth']
            if 'stock_hour6_22_cnt' in df_truth.columns:
                truth_cols.append('stock_hour6_22_cnt')

            merged_df = pd.merge(
                df_pred_with_ids,
                df_truth[truth_cols],
                on=['store_id', 'product_id', 'date'],
                how='inner'
            )
            # save merge_df as parquet
            eval_model_name = self.args.des.replace('recovered_', '')
            self._calculate_and_save_metrics(
                merged_df=merged_df,
                data_split="train",
                eval_model_name=eval_model_name
            )

        return

    def evaluation_vs_recovered(self, df_pred, recovered_eval_path):
        """Compares DLinear predictions against recovered demand from the eval set."""
        print("\n--- Evaluating Forecast vs. Recovered Demand (on Eval Set) ---")
        try:
            recovered_eval_df = pd.read_parquet(recovered_eval_path)
            # DLinear uses 'date' column, align it
            recovered_eval_df = recovered_eval_df.rename(columns={'dt': 'date'})
        except FileNotFoundError:
            print(f"Warning: Recovered eval file not found at {recovered_eval_path}. Skipping this evaluation.")
            return
        
        # Extract the recovery model name from the evaluation path
        eval_model_name = os.path.basename(recovered_eval_path).replace('demand_eval_', '').replace('.parquet', '')

        # --- Gemini Code Assist: Fix merge error by ensuring consistent key types ---
        # Convert 'date' columns in both dataframes to string format 'YYYY-MM-DD' before merging.
        # This prevents the "You are trying to merge on datetime64[ns] and object columns" error.
        df_pred['date'] = pd.to_datetime(df_pred['date']).dt.strftime('%Y-%m-%d')
        recovered_eval_df['date'] = pd.to_datetime(recovered_eval_df['date']).dt.strftime('%Y-%m-%d')
        recovered_eval_df = recovered_eval_df.rename(columns={'sale_amount_pred': 'recovered_demand_truth'})

        # Merge forecast predictions with recovered demand ground truth
        merged_df = pd.merge(
            df_pred,
            recovered_eval_df[['store_id', 'product_id', 'date', 'recovered_demand_truth']],
            on=['store_id', 'product_id', 'date']
        )
        
        self._calculate_and_save_metrics(merged_df, 'eval', eval_model_name)

    # def data_organization_from_df_source(self, np_pred, df_source):
    #     """
    #     Organizes raw numpy predictions into a structured DataFrame with dates and IDs.
    #     This version is adapted to work for both train and eval splits.
    #     """
    #     group_ids = ['store_id', 'product_id']
        
    #     # The number of sequences generated by the dataloader is len(df) - total_len + 1
    #     # The predictions correspond to the end of each of these sequences.
    #     num_sequences = len(df_source) - self.args.total_seq_len + 1
        
    #     # The predictions array should match this size
    #     assert len(np_pred) == num_sequences, f"Prediction array size mismatch: {len(np_pred)} vs {num_sequences}"

    #     # Create a dataframe to hold the predictions
    #     all_preds = []
    #     for i in range(num_sequences):
    #         # The prediction `i` corresponds to the sequence starting at index `i` in `df_source`.
    #         # The prediction starts after `seq_len` days.
    #         start_idx = i + self.args.seq_len
    #         end_idx = start_idx + self.args.pred_len
            
    #         # Get the corresponding slice from the source dataframe for IDs and dates
    #         ref_slice = df_source.iloc[start_idx:end_idx]
            
    #         pred_df_slice = ref_slice[group_ids + ['date']].copy()
    #         pred_df_slice['sale_amount_pred'] = np_pred[i].squeeze()
    #         all_preds.append(pred_df_slice)
            
    #     return pd.concat(all_preds, ignore_index=True).drop_duplicates()

    def cal_metrics(self, df, target_dates, groups=["psd>=0"]):
        sample_cnt, preds_sum, actuals_sum, preds_mean, actuals_mean = {}, {}, {}, {}, {}
        acc, wape, wpe, mae, bias = {}, {}, {}, {}, {}
        res = pd.DataFrame()
        for group in groups:
            df1 = df.query(group)
            sub_res = pd.DataFrame(target_dates, columns=["date"])
            for target_date in target_dates:
                df2 = df1[(df1.date == target_date) &
                          (df1.stock_hour6_22_cnt == 0)]
                preds = df2.prediction
                actuals = df2.sale_amount
                # import pdb; pdb.set_trace()
                sample_cnt[target_date] = len(df2)
                # print("len_df2", len(df2))
                preds_sum[target_date] = preds.sum(axis=0)
                actuals_sum[target_date] = actuals.sum(axis=0)
                preds_mean[target_date] = preds.mean(axis=0)
                actuals_mean[target_date] = actuals.mean(axis=0)
                acc[target_date] = 1 - \
                    (preds - actuals).abs().sum(axis=0) / \
                    actuals.abs().sum(axis=0)
                wape[target_date] = (
                    preds - actuals).abs().sum(axis=0) / actuals.abs().sum(axis=0)
                wpe[target_date] = (
                    preds - actuals).sum(axis=0) / actuals.abs().sum(axis=0)
                mae[target_date] = (preds - actuals).abs().mean(axis=0)
                bias[target_date] = (preds - actuals).mean(axis=0)
            sub_res['sample_cnt'] = sample_cnt.values()
            sub_res['preds_sum'] = preds_sum.values()
            sub_res['actuals_sum'] = actuals_sum.values()
            sub_res['preds_mean'] = preds_mean.values()
            sub_res['actuals_mean'] = actuals_mean.values()
            sub_res['acc'] = acc.values()
            sub_res['wape'] = wape.values()
            sub_res['wpe'] = wpe.values()
            sub_res['mae'] = mae.values()
            sub_res['bias'] = bias.values()
            sub_res.insert(0, 'group_type', group)
            # sub_res.insert(0, 'data_type', data_type)
            res = pd.concat([res, sub_res], ignore_index=True)
        return res

