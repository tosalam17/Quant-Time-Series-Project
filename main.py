# =============================================================================
# GTSF Quant Mentorship S26 — Time-Series Forecasting of U.S. Equities
# Models: ARIMA(1,1,1) · LSTM (PyTorch) · N-HiTS (NeuralForecast)
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

from copy import deepcopy as dc
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE


# =============================================================================
# PART 1: DATA COLLECTION, CLEANING & EDA
# =============================================================================

def load_data():
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    tsf_data = yf.download(tickers, start="2007-01-01", end="2025-12-31")

    print(f"Start: {tsf_data.index.min()}, End: {tsf_data.index.max()}")
    print(f"Total trading days: {len(tsf_data)}")
    print(f"Null values:\n{tsf_data.isna().sum()}")
    print(f"Duplicated dates: {tsf_data.index.duplicated().sum()}")

    # Check for suspicious date gaps (> weekend)
    date_gaps = tsf_data.index.to_series().diff().dt.days
    suspicious = date_gaps[date_gaps > 5]
    if len(suspicious) > 0:
        print(f"Suspicious date gaps:\n{suspicious}")

    return tsf_data, tickers


def flag_outliers(tsf_data):
    log_returns = np.log(tsf_data['Close'] / tsf_data['Close'].shift(1))
    outlier_flags = log_returns.abs() > 0.10
    outlier_flags.columns = pd.MultiIndex.from_tuples(
        [('Outlier', col) for col in outlier_flags.columns]
    )
    tsf_data = pd.concat([tsf_data, outlier_flags], axis=1)
    return tsf_data, log_returns


def plot_outliers_per_year(tsf_data):
    outlier_counts = tsf_data['Outlier'].copy()
    outlier_counts.index = tsf_data.index.year
    yearly_outliers = outlier_counts.groupby(level=0).sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    yearly_outliers.plot(kind='bar', ax=ax, width=0.7)
    ax.set_title('Number of Outlier Days (>10% Single-Day Move) Per Year', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Outlier Days')
    ax.legend(title='Ticker')
    ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='2008 Crisis')
    ax.axvline(x=13.5, color='orange', linestyle='--', alpha=0.5, label='COVID')
    plt.tight_layout()
    plt.show()

    year_out_c = yearly_outliers.copy()
    year_out_c["Count"] = yearly_outliers.sum(axis=1)
    print(year_out_c.sort_values(by='Count', ascending=False))


def plot_prices_and_returns(tsf_data, log_returns):
    normalized = tsf_data['Close'] / tsf_data['Close'].iloc[0] * 100
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    normalized.plot(ax=axes[0])
    axes[0].set_title('Normalized Closing Prices (Base = 100, Jan 2007)', fontsize=14)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Normalized Price')
    axes[0].legend(title='Ticker')
    axes[0].axvspan('2008-09-01', '2009-06-01', color='red', alpha=0.1, label='2008 Crisis')
    axes[0].axvspan('2020-02-01', '2020-04-01', color='orange', alpha=0.1, label='COVID')

    log_returns.plot(ax=axes[1], alpha=0.7)
    axes[1].set_title('Daily Log Returns', fontsize=14)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Log Return')
    axes[1].legend(title='Ticker')
    axes[1].axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    plt.show()


def plot_rolling_volatility(log_returns):
    rolling_vol = log_returns.rolling(window=30).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(14, 5))
    rolling_vol.plot(ax=ax)
    ax.set_title('30-Day Rolling Volatility (Annualized)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.legend(title='Ticker')
    ax.axvspan('2008-09-01', '2009-06-01', color='red', alpha=0.1)
    ax.axvspan('2020-02-01', '2020-04-01', color='orange', alpha=0.1)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(log_returns):
    fig, ax = plt.subplots(figsize=(7, 5))
    corr_matrix = log_returns.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title('Correlation Matrix of Daily Log Returns (2007-2025)', fontsize=14)
    plt.tight_layout()
    plt.show()


# =============================================================================
# PART 2: MODEL 1 — ARIMA(1,1,1)
# =============================================================================

def run_arima(tsf_data, tickers):
    print("\n" + "="*60)
    print("MODEL 1: ARIMA(1,1,1)")
    print("="*60)

    # Stationarity check on raw prices
    print("\nADF test on raw prices:")
    for tick in tickers:
        series = tsf_data["Close"][tick]
        print(f"  {tick} p-val: {adfuller(series)[1]:.4f}")

    # Transform to log returns and re-test
    arima_data = {}
    print("\nADF test on log returns:")
    for tick in tickers:
        series = np.log(tsf_data['Close'][tick] / tsf_data['Close'][tick].shift(1)).dropna()
        series.index = pd.DatetimeIndex(series.index)
        series = series.asfreq('B', fill_value=0)
        arima_data[tick] = series
        result = adfuller(arima_data[tick])
        print(f"  {tick} ADF: {result[0]:.4f}, p-value: {result[1]:.6f}")

    # ACF/PACF plots
    for tick in tickers:
        plot_acf(arima_data[tick], title=f'{tick} ACF')
        plot_pacf(arima_data[tick], title=f'{tick} PACF')
        plt.show()

    # Train/test split — 80/20, test capped at 200 days
    split = int(len(arima_data['AAPL']) * 0.8)
    arima_train = {tick: arima_data[tick][:split] for tick in tickers}
    arima_test  = {tick: arima_data[tick][split:split+200] for tick in tickers}
    print(f"\nTrain size: {split}, Test size: 200")

    # Fit ARIMA(1,1,1)
    arima_models = {}
    arima_results = {}
    for tick in tickers:
        model = ARIMA(arima_train[tick], order=(1, 1, 1))
        result = model.fit(method_kwargs={"maxiter": 1000})
        arima_models[tick] = model
        arima_results[tick] = result
        print(f"  {tick} AIC: {result.aic:.4f}, BIC: {result.bic:.4f}")

    # Residual plots
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for i, tick in enumerate(tickers):
        residuals = pd.Series(arima_results[tick].resid[1:])
        residuals.plot(title=f'{tick} Residuals', ax=axes[i][0])
        residuals.plot(title=f'{tick} Density', kind='kde', ax=axes[i][1])
    plt.tight_layout()
    plt.show()

    # Forecast plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    for i, tick in enumerate(tickers):
        forecast = arima_results[tick].forecast(steps=len(arima_test[tick]))
        axes[i].plot(arima_test[tick].index, arima_test[tick], label='Actual', color='orange')
        axes[i].plot(arima_test[tick].index, forecast, label='Forecast', color='red', linestyle='--')
        axes[i].set_title(f'{tick} ARIMA(1,1,1) Forecast vs Actual')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Log Return')
        axes[i].legend()
    plt.tight_layout()
    plt.show()

    # Evaluation metrics
    arima_metrics = {}
    print("\nARIMA Metrics:")
    for tick in tickers:
        forecast_log = arima_results[tick].forecast(steps=len(arima_test[tick]))
        rmse = np.sqrt(mean_squared_error(arima_test[tick], forecast_log))
        mae  = mean_absolute_error(arima_test[tick], forecast_log)
        arima_metrics[tick] = {'RMSE': rmse, 'MAE': mae}
        print(f"  {tick} -> RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    return arima_results, arima_test, arima_metrics


# =============================================================================
# PART 2: MODEL 2 — LSTM (PyTorch)
# =============================================================================

def preparing_lstm_df(df, n_steps, ticker):
    data = pd.DataFrame()
    data['Close'] = np.log(dc(df['Close'][ticker]) / dc(df['Close'][ticker]).shift(1))
    data.dropna(inplace=True)
    for i in range(1, n_steps + 1):
        data[f'Close(t-{i})'] = data['Close'].shift(i)
    data.dropna(inplace=True)
    return data


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden1=64, hidden2=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc    = nn.Linear(hidden2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out


def run_lstm(tsf_data, tickers, split):
    print("\n" + "="*60)
    print("MODEL 2: LSTM (PyTorch)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Device: {device}")

    n_steps = 7
    lstm_data = {tick: preparing_lstm_df(tsf_data, n_steps, tick) for tick in tickers}

    train_lstms = {tick: lstm_data[tick][:split] for tick in tickers}
    test_lstms  = {tick: lstm_data[tick][split:split+200] for tick in tickers}

    # Scale per ticker
    scalers           = {}
    lstm_train_scaled = {}
    lstm_test_scaled  = {}
    for tick in tickers:
        scalers[tick]           = MinMaxScaler(feature_range=(-1, 1))
        lstm_train_scaled[tick] = scalers[tick].fit_transform(train_lstms[tick].to_numpy())
        lstm_test_scaled[tick]  = scalers[tick].transform(test_lstms[tick].to_numpy())

    X_train_vals = {tick: lstm_train_scaled[tick][:, 1:] for tick in tickers}
    y_train_vals = {tick: lstm_train_scaled[tick][:, 0]  for tick in tickers}
    X_test_vals  = {tick: lstm_test_scaled[tick][:, 1:]  for tick in tickers}
    y_test_vals  = {tick: lstm_test_scaled[tick][:, 0]   for tick in tickers}

    X_train_vals = {tick: X_train_vals[tick].reshape(*X_train_vals[tick].shape, 1) for tick in tickers}
    X_test_vals  = {tick: X_test_vals[tick].reshape(*X_test_vals[tick].shape, 1)   for tick in tickers}

    # Tensors and DataLoaders
    X_train_tensors  = {tick: torch.tensor(X_train_vals[tick], dtype=torch.float32) for tick in tickers}
    y_train_tensors  = {tick: torch.tensor(y_train_vals[tick], dtype=torch.float32) for tick in tickers}
    train_dataframes = {tick: TensorDataset(X_train_tensors[tick], y_train_tensors[tick]) for tick in tickers}
    train_loaders    = {tick: DataLoader(train_dataframes[tick], batch_size=32, shuffle=False) for tick in tickers}

    lstm_models  = {tick: LSTMModel().to(device) for tick in tickers}
    criterion    = nn.L1Loss()
    optimizers   = {tick: torch.optim.Adam(lstm_models[tick].parameters()) for tick in tickers}

    # Training
    n_epochs  = 300
    patience  = 50
    best_losses = {}

    for tick in tickers:
        print(f"\nTraining LSTM for {tick}...")
        torch.manual_seed(42)
        np.random.seed(42)
        best_loss   = float('inf')
        patience_cnt = 0
        best_weights = {k: v.clone() for k, v in lstm_models[tick].state_dict().items()}

        for epoch in range(n_epochs):
            lstm_models[tick].train()
            train_loss = 0
            for X_batch, y_batch in train_loaders[tick]:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizers[tick].zero_grad()
                preds = lstm_models[tick](X_batch).squeeze()
                loss  = criterion(preds, y_batch.squeeze())
                loss.backward()
                optimizers[tick].step()
                train_loss += loss.item()
            train_loss /= len(train_loaders[tick])

            if train_loss < best_loss:
                best_loss    = train_loss
                patience_cnt = 0
                best_weights = {k: v.clone() for k, v in lstm_models[tick].state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} | Loss: {train_loss:.6f}")

        best_losses[tick] = best_loss
        lstm_models[tick].load_state_dict(best_weights)
        print(f"  {tick} training complete — best loss: {best_loss:.6f}")

    # Predict and inverse-transform
    lstm_predictions = {}
    y_test_actual    = {}
    for tick in tickers:
        lstm_models[tick].eval()
        with torch.no_grad():
            X_test_t     = torch.tensor(X_test_vals[tick], dtype=torch.float32).to(device)
            y_pred_scaled = lstm_models[tick](X_test_t).cpu().numpy()

        dummy_pred = np.zeros((len(y_pred_scaled),    lstm_train_scaled[tick].shape[1]))
        dummy_test = np.zeros((len(y_test_vals[tick]), lstm_train_scaled[tick].shape[1]))
        dummy_pred[:, 0] = y_pred_scaled.squeeze()
        dummy_test[:, 0] = y_test_vals[tick]

        lstm_predictions[tick] = scalers[tick].inverse_transform(dummy_pred)[:, 0]
        y_test_actual[tick]    = scalers[tick].inverse_transform(dummy_test)[:, 0]

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    for i, tick in enumerate(tickers):
        axes[i].plot(y_test_actual[tick],    label='Actual',    color='blue')
        axes[i].plot(lstm_predictions[tick], label='Predicted', color='orange')
        axes[i].set_title(f'{tick} LSTM Predictions vs Actual')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Log Return')
        axes[i].legend()
    plt.tight_layout()
    plt.show()

    # Metrics
    lstm_metrics = {}
    print("\nLSTM Metrics:")
    for tick in tickers:
        rmse = np.sqrt(mean_squared_error(y_test_actual[tick], lstm_predictions[tick]))
        mae  = mean_absolute_error(y_test_actual[tick], lstm_predictions[tick])
        lstm_metrics[tick] = {'RMSE': rmse, 'MAE': mae}
        print(f"  {tick} — RMSE: {rmse:.6f} | MAE: {mae:.6f}")

    return lstm_predictions, y_test_actual, lstm_metrics


# =============================================================================
# PART 2: MODEL 3 — N-HiTS (NeuralForecast)
# =============================================================================

def prepare_nhits_df(tsf_data, ticker):
    df = pd.DataFrame({
        'unique_id': ticker,
        'ds': dc(tsf_data['Close'][ticker].index),
        'y':  np.log(dc(tsf_data['Close'][ticker]) / dc(tsf_data['Close'][ticker]).shift(1))
    }).dropna()
    return df


def run_nhits(tsf_data, tickers):
    print("\n" + "="*60)
    print("MODEL 3: N-HiTS (NeuralForecast)")
    print("="*60)

    nhits_dfs  = {tick: prepare_nhits_df(tsf_data, tick) for tick in tickers}
    nhits_data = pd.concat(nhits_dfs.values(), ignore_index=True)

    cutoff     = nhits_data['ds'].quantile(0.8)
    nhits_train = nhits_data[nhits_data['ds'] < cutoff]
    nhits_test  = nhits_data[nhits_data['ds'] >= cutoff].groupby('unique_id').head(200)
    horizon     = 200

    nhits_models      = {}
    nhits_predictions = {}

    for tick in tickers:
        print(f"\nTraining N-HiTS for {tick}...")
        ticker_train = nhits_train[nhits_train['unique_id'] == tick]

        nf = NeuralForecast(
            models=[NHITS(
                h=horizon,
                input_size=horizon,
                loss=MAE(),
                max_steps=1000,
                batch_size=32,
                random_seed=42,
                start_padding_enabled=False,
                val_check_steps=50
            )],
            freq='B'
        )
        nf.fit(ticker_train)
        preds = nf.predict()

        nhits_models[tick]      = nf
        nhits_predictions[tick] = preds
        print(f"  {tick} training complete")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    for i, tick in enumerate(tickers):
        actual = nhits_test[nhits_test['unique_id'] == tick]
        preds  = nhits_predictions[tick]
        axes[i].plot(actual['ds'].values,    actual['y'].values,   label='Actual',   color='blue')
        axes[i].plot(preds['ds'].values,     preds['NHITS'].values, label='Forecast', color='orange')
        axes[i].set_title(f'{tick} N-HiTS Forecast vs Actual')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Log Return')
        axes[i].legend()
    plt.tight_layout()
    plt.show()

    # Metrics
    nhits_metrics = {}
    print("\nN-HiTS Metrics:")
    for tick in tickers:
        y_true = nhits_test[nhits_test['unique_id'] == tick]['y'].values
        y_pred = nhits_predictions[tick]['NHITS'].values
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        nhits_metrics[tick] = {'RMSE': rmse, 'MAE': mae}
        print(f"  {tick} -> RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    return nhits_predictions, nhits_test, nhits_metrics


# =============================================================================
# PART 2: MODEL COMPARISON TABLE
# =============================================================================

def print_comparison_table(arima_metrics, lstm_metrics, nhits_metrics, tickers):
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    comparison_data = {
        'Model':  ['ARIMA']*4 + ['LSTM']*4 + ['N-HiTS']*4,
        'Ticker': tickers * 3,
        'RMSE': [arima_metrics[t]['RMSE'] for t in tickers] +
                [lstm_metrics[t]['RMSE']  for t in tickers] +
                [nhits_metrics[t]['RMSE'] for t in tickers],
        'MAE':  [arima_metrics[t]['MAE']  for t in tickers] +
                [lstm_metrics[t]['MAE']   for t in tickers] +
                [nhits_metrics[t]['MAE']  for t in tickers],
    }
    comparison_df = pd.DataFrame(comparison_data)
    pivot_rmse = comparison_df.pivot(index='Ticker', columns='Model', values='RMSE')
    pivot_mae  = comparison_df.pivot(index='Ticker', columns='Model', values='MAE')

    print("\nRMSE Comparison (lower is better):")
    print(pivot_rmse[['ARIMA', 'LSTM', 'N-HiTS']].round(6).to_string())
    print("\nMAE Comparison (lower is better):")
    print(pivot_mae[['ARIMA', 'LSTM', 'N-HiTS']].round(6).to_string())

    print("\nBest Model Per Ticker (by RMSE):")
    for tick in tickers:
        row  = pivot_rmse.loc[tick]
        best = row.idxmin()
        print(f"  {tick}: {best} (RMSE: {row[best]:.6f})")


# =============================================================================
# PART 3: DIEBOLD-MARIANO TEST
# =============================================================================

def diebold_mariano_test(e1, e2, loss='squared'):
    """
    H0: Model 1 and Model 2 have equal predictive accuracy.
    Positive DM stat -> model 2 more accurate.
    Negative DM stat -> model 1 more accurate.
    """
    d = (e1**2 - e2**2) if loss == 'squared' else (np.abs(e1) - np.abs(e2))
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1) / len(d)
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value


def run_diebold_mariano(arima_results, arima_test, y_test_actual,
                        lstm_predictions, nhits_test, nhits_predictions, tickers):
    print("\n" + "="*60)
    print("PART 3: DIEBOLD-MARIANO TEST")
    print("="*60)

    arima_errors = {}
    lstm_errors  = {}
    nhits_errors = {}

    for tick in tickers:
        arima_forecast      = arima_results[tick].forecast(steps=200).values
        arima_errors[tick]  = arima_test[tick].values - arima_forecast
        lstm_errors[tick]   = y_test_actual[tick] - lstm_predictions[tick]
        nhits_errors[tick]  = (
            nhits_test[nhits_test['unique_id'] == tick]['y'].values[:200]
            - nhits_predictions[tick]['NHITS'].values[:200]
        )

    print("\nTest 1: ARIMA vs LSTM (H0: equal accuracy)")
    for tick in tickers:
        dm, p = diebold_mariano_test(arima_errors[tick], lstm_errors[tick])
        sig   = "✅ significant" if p < 0.05 else "❌ not significant"
        print(f"  {tick}: DM = {dm:.4f}, p = {p:.4f} → {sig}")

    print("\nTest 2: LSTM vs N-HiTS (H0: equal accuracy)")
    for tick in tickers:
        dm, p = diebold_mariano_test(lstm_errors[tick], nhits_errors[tick])
        sig   = "✅ significant" if p < 0.05 else "❌ not significant"
        print(f"  {tick}: DM = {dm:.4f}, p = {p:.4f} → {sig}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- Data ---
    tsf_data, tickers = load_data()
    tsf_data, log_returns = flag_outliers(tsf_data)

    # --- EDA ---
    plot_outliers_per_year(tsf_data)
    plot_prices_and_returns(tsf_data, log_returns)
    plot_rolling_volatility(log_returns)
    plot_correlation_heatmap(log_returns)

    # Shared train/test split index
    split = int(len(log_returns) * 0.8)

    # --- Models ---
    arima_results, arima_test, arima_metrics = run_arima(tsf_data, tickers)
    lstm_predictions, y_test_actual, lstm_metrics = run_lstm(tsf_data, tickers, split)
    nhits_predictions, nhits_test, nhits_metrics = run_nhits(tsf_data, tickers)

    # --- Comparison ---
    print_comparison_table(arima_metrics, lstm_metrics, nhits_metrics, tickers)

    # --- Statistical Test ---
    run_diebold_mariano(
        arima_results, arima_test,
        y_test_actual, lstm_predictions,
        nhits_test, nhits_predictions,
        tickers
    )