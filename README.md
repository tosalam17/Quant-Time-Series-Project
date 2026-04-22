# Time-Series Forecasting of U.S. Equities

### GTSF Quant Mentorship S26 \- Final Project

A end-to-end time-series forecasting pipeline comparing three models: **ARIMA**, **LSTM**, and **N-HiTS**, on daily log returns of four large-cap U.S. equities, with statistical model comparison via the Diebold-Mariano test.

---

## Overview

|  |  |
| :---- | :---- |
| **Tickers** | AAPL, GOOGL, MSFT, NVDA |
| **Data** | Daily OHLCV, Jan 2007 – Dec 2024 (\~4,500 trading days) |
| **Target** | Log returns: r\_t \= ln(P\_t / P\_{t-1}) |
| **Evaluation window** | 200-day out-of-sample test set (identical across all models) |
| **Metrics** | RMSE, MAE on log returns |

---

## Models

### ARIMA(1,1,1): Baseline

- Order selected via ACF/PACF analysis of log returns (significant spike at lag 1\)  
- Stationarity confirmed via ADF test after log differencing  
- Converges to near-zero flat forecast, consistent with the Efficient Market Hypothesis

### LSTM (PyTorch)

- Two-layer LSTM: hidden sizes 64 → 32 → fully connected output  
- 7-day lookback window, per-ticker MinMaxScaler to prevent data leakage  
- Adam optimizer, MAE loss, early stopping (patience=50), 300 max epochs  
- Random seed 42 per ticker for reproducibility

### N-HiTS (NeuralForecast)

- Separate model per ticker, 3.1M parameters across 3 stacked MLP blocks  
- Horizon \= 200 business days, input\_size \= 200, MAE loss, 1,000 gradient steps  
- Trained on Apple MPS (GPU)

---

## Results

### Evaluation Metrics (200-Day Test Window)

| Ticker | ARIMA RMSE | ARIMA MAE | LSTM RMSE | LSTM MAE | N-HiTS RMSE | N-HiTS MAE |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| AAPL | 0.022981 | 0.017505 | **0.018104** | **0.013252** | 0.024415 | 0.019099 |
| GOOGL | 0.024651 | 0.018798 | 0.025951 | 0.018793 | 0.025914 | 0.019842 |
| MSFT | 0.022439 | 0.017015 | 0.021993 | 0.016170 | 0.023916 | 0.018249 |
| NVDA | 0.039297 | 0.031005 | 0.038651 | 0.027819 | 0.044696 | 0.035477 |

### Diebold-Mariano Test (α \= 0.05)

| Ticker | DM (ARIMA vs LSTM) | p-value | DM (LSTM vs N-HiTS) | p-value |
| :---- | :---- | :---- | :---- | :---- |
| AAPL | \+2.47 | **0.014 ✓** | −3.05 | **0.002 ✓** |
| GOOGL | −0.51 | 0.608 ✗ | \+0.01 | 0.988 ✗ |
| MSFT | \+0.21 | 0.830 ✗ | −0.92 | 0.357 ✗ |
| NVDA | \+0.15 | 0.883 ✗ | −1.40 | 0.163 ✗ |

Only 2 of 8 comparisons are statistically significant, both on AAPL, both favoring LSTM.

---

## Key Findings

1. **EMH confirmed**: Model complexity does not reliably improve forecasts on daily log returns. 6 of 8 DM comparisons are statistically indistinguishable from noise.  
2. **LSTM wins on AAPL**: The only ticker with a proven, statistically significant edge for LSTM over both ARIMA and N-HiTS.  
3. **Architecture mismatch**: N-HiTS (3.1M params) underperforms LSTM (\~16K params) on every ticker. Its hierarchical decomposition finds little signal in near-random log returns.  
4. **NVDA is hardest to forecast**: Highest errors across all three models due to idiosyncratic AI/GPU-sector volatility.

---

## Repository Structure

├── quant-tsf.ipynb          \# Main notebook (data, EDA, modeling, evaluation)

├── GTSF\_TSF\_Writeup.docx    \# Summary analysis report

├── GTSF\_TSF\_Slide.pptx      \# Presentation slide

└── README.md

---

## Setup

\# Clone the repo

git clone https://github.com/\<your-username\>/\<repo-name\>.git

cd \<repo-name\>

\# Install dependencies

pip install yfinance pandas numpy matplotlib seaborn statsmodels

pip install torch

pip install neuralforecast

pip install scipy scikit-learn

---

## Data Source

Data collected via [yfinance](https://github.com/ranaroussi/yfinance) \- free, no API key required.

import yfinance as yf

data \= yf.download(\["AAPL", "MSFT", "GOOGL", "NVDA"\], start="2007-01-01", end="2024-12-31")

---

*GTSF Quant Mentorship S26 · Toye Salami · 2026*  
