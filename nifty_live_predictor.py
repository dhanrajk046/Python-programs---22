"""
nifty_live_predictor.py

Fetches Nifty data from Yahoo Finance (^NSEI by default), computes indicators,
trains a simple RandomForest to predict next-period direction, and runs
a live loop to produce predictions using both the ML model and indicator filters.

Drop-in-use in VS Code. Replace ticker with your futures data source if available.
"""

import time
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# -----------------------
# USER PARAMETERS
# -----------------------
TICKER = "^NSEI"       # '^NSEI' = Nifty 50 index on Yahoo. Replace if you have futures ticker.
INTERVAL = "5m"        # data interval for "live" (1m,2m,5m,15m,...). Use what your data allows.
HIST_PERIOD = "180d"   # history period to train on (adjust as needed)
MODEL_PATH = "nifty_model.joblib"
PREDICTION_HORIZON = 1  # predict next bar direction (1 bar ahead)
MIN_TRAIN_ROWS = 200    # minimum rows required to train
LIVE_LOOP_INTERVAL = 60  # seconds between live checks (adjust as needed)

# -----------------------
# UTILITY FUNCTIONS
# -----------------------

def fetch_history(ticker=TICKER, period=HIST_PERIOD, interval=INTERVAL):
    """Fetch historical OHLC data using yfinance"""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("Empty data returned from yfinance for ticker: {}".format(ticker))
    df = df.dropna()
    return df

def compute_indicators(df):
    """Compute top 5 indicator filters + extra features"""

    df = df.copy()

    # EMAs / SMA (trend)
    df["EMA50"] = ta.ema(df["Close"], length=50)
    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["SMA200"] = ta.sma(df["Close"], length=200)

    # RSI
    df["RSI14"] = ta.rsi(df["Close"], length=14)

    # MACD (macd, macd_signal, macd_hist)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]
    df["MACD_HIST"] = macd["MACDh_12_26_9"]

    # ATR (volatility)
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Supertrend implementation using pandas_ta
    st = ta.supertrend(df["High"], df["Low"], df["Close"], length=10, multiplier=3.0)
    # pandas_ta returns columns: SUPERT_10_3.0 and SUPERTd_10_3.0 (direction)
    # rename for convenience
    df["SUPERT"] = st[f"SUPERT_10_3.0"]
    df["SUPERT_DIR"] = st[f"SUPERTd_10_3.0"]  # 1 = up, -1 = down

    # Additional features: returns, volume change, candle size
    df["RET"] = df["Close"].pct_change()
    df["VOL_CHG"] = df["Volume"].pct_change()
    df["CANDLE_BODY"] = (df["Close"] - df["Open"]) / df["High"] - df["Low"].replace(0, np.nan)

    # Clean up: forward/backfill needed indicator NaNs
    df = df.dropna()
    return df

def build_label(df, horizon=PREDICTION_HORIZON):
    """Label: 1 if future close > current close, else 0. (Binary direction)"""
    df = df.copy()
    df["FUT_CLOSE"] = df["Close"].shift(-horizon)
    df = df.dropna()
    df["TARGET"] = (df["FUT_CLOSE"] > df["Close"]).astype(int)
    return df

# -----------------------
# SIGNAL / FILTERS (Institutional style)
# -----------------------

def indicator_filters_signal(row):
    """
    Apply institutional-style filters:
    - Trend filter: EMA50 > EMA200 (bull) or EMA50 < EMA200 (bear)
    - RSI filter: RSI14 > 50 for bullish bias
    - MACD filter: MACD_HIST > 0
    - ATR filter: ATR14 relative threshold (we use normalized ATR to recent price)
    - Supertrend direction: SUPERT_DIR == 1 (bull) or -1 (bear)

    Returns a score from -5 .. +5 and textual explanation.
    """
    score = 0
    reasons = []

    # Trend
    if row["EMA50"] > row["EMA200"]:
        score += 1
        reasons.append("Trend: EMA50>EMA200")
    else:
        score -= 1
        reasons.append("Trend: EMA50<EMA200")

    # RSI
    if row["RSI14"] > 55:  # slightly stricter than 50 for better signal
        score += 1
        reasons.append(f"Momentum: RSI={row['RSI14']:.1f}>55")
    elif row["RSI14"] < 45:
        score -= 1
        reasons.append(f"Momentum: RSI={row['RSI14']:.1f}<45")

    # MACD hist
    if row["MACD_HIST"] > 0:
        score += 1
        reasons.append("MACD hist > 0")
    else:
        score -= 1
        reasons.append("MACD hist <= 0")

    # ATR normalized (higher ATR = more volatile). For filter we'll prefer moderate ATR but not too low.
    atr_ratio = row["ATR14"] / row["Close"]
    if 0.0008 < atr_ratio < 0.02:  # thresholds depend on interval; tune for your timeframe
        score += 1
        reasons.append(f"ATR ok ({atr_ratio:.5f})")
    else:
        # if very low volatility penalize; extremely high also penalize
        score -= 1
        reasons.append(f"ATR out ({atr_ratio:.5f})")

    # Supertrend
    if row["SUPERT_DIR"] == 1:
        score += 1
        reasons.append("Supertrend = UP")
    else:
        score -= 1
        reasons.append("Supertrend = DOWN")

    return score, "; ".join(reasons)

# -----------------------
# MODEL: train / predict
# -----------------------

def prepare_features(df):
    """Select feature columns for ML model"""
    features = [
        "Close", "EMA50", "EMA200", "RSI14", "MACD", "MACD_SIGNAL", "MACD_HIST",
        "ATR14", "SUPERT", "SUPERT_DIR", "RET", "VOL_CHG"
    ]
    X = df[features].copy()
    # Add engineered ratios
    X["EMA50_EMA200_DIFF"] = X["EMA50"] - X["EMA200"]
    X["MACD_HIST_OVER_ATR"] = X["MACD_HIST"] / X["ATR14"].replace(0, np.nan)
    X = X.fillna(method="ffill").fillna(0)
    return X

def train_model(df):
    """Train a RandomForest classifier and save it."""
    df = build_label(df)
    X = prepare_features(df)
    y = df["TARGET"]

    if len(X) < MIN_TRAIN_ROWS:
        raise RuntimeError("Not enough rows to train. Need at least {}".format(MIN_TRAIN_ROWS))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print("Model evaluation on hold-out set:")
    print("Accuracy:", accuracy_score(y_test, preds))
    try:
        print("ROC AUC:", roc_auc_score(y_test, probas))
    except Exception:
        pass
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)
    return model

# -----------------------
# LIVE PREDICTION LOOP
# -----------------------

def load_or_train_model(df):
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded model from", MODEL_PATH)
        return model
    except Exception:
        print("No saved model found — training new model.")
        return train_model(df)

def predict_row(model, row):
    """Predict using model and compute filter score to combine outputs."""
    Xrow = prepare_features(pd.DataFrame([row]))  # prepare_features expects df
    prob = model.predict_proba(Xrow)[:, 1][0]
    pred = int(prob > 0.5)
    filt_score, reasons = indicator_filters_signal(row)
    # Combine model prob with filter score into a final confidence metric
    # Normalize filt_score (-5..5) to 0..1
    filt_norm = (filt_score + 5) / 10
    combined_confidence = 0.6 * prob + 0.4 * filt_norm  # weigh model more
    # Decide final direction: require agreement or high confidence
    if combined_confidence > 0.6:
        final_dir = "LONG"
    elif combined_confidence < 0.4:
        final_dir = "SHORT"
    else:
        final_dir = "NEUTRAL"

    return {
        "model_prob": prob,
        "filter_score": filt_score,
        "filter_reasons": reasons,
        "combined_confidence": combined_confidence,
        "final_dir": final_dir,
        "pred_label": pred
    }

def run_live_loop():
    """Main loop to fetch latest bar, compute indicators, and print prediction"""
    print("Fetching history for training...")
    hist = fetch_history()
    hist = compute_indicators(hist)
    # Train or load model
    model = load_or_train_model(hist)

    print("Entering live prediction loop. Press Ctrl+C to stop.")
    try:
        while True:
            # fetch the most recent few bars (short period)
            latest = fetch_history(period="5d")  # 5 days worth of latest bars
            latest = compute_indicators(latest)
            latest = latest.sort_index()

            last_row = latest.iloc[-1]
            pred = predict_row(model, last_row)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("="*60)
            print(f"[{now}] Ticker={TICKER} Interval={INTERVAL}")
            print(f"Close: {last_row['Close']:.2f} | Model Prob(next up): {pred['model_prob']:.3f}")
            print(f"Indicator Score: {pred['filter_score']} ({pred['filter_reasons']})")
            print(f"Combined Confidence: {pred['combined_confidence']:.3f} => FINAL: {pred['final_dir']}")
            print("="*60)

            # Wait, then fetch again
            time.sleep(LIVE_LOOP_INTERVAL)
    except KeyboardInterrupt:
        print("Live loop interrupted by user. Exiting.")

# -----------------------
# BACKTEST HELPER (quick look)
# -----------------------
def quick_backtest(df, model):
    """Simple backtest: apply model predictions to test set and compute directional stats."""
    df_label = build_label(df)
    X = prepare_features(df_label)
    y = df_label["TARGET"]
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    print("Quick backtest accuracy (full sample):", acc)
    # add returns: assume 1 unit per correct direction trade
    returns = ( (preds == y).astype(int) * 1.0 ) - ( (preds != y).astype(int) * 0.5 )  # simplistic P/L metric
    print("Approx P/L metric (sum):", returns.sum())

# -----------------------
# RUN AS SCRIPT
# -----------------------
if __name__ == "__main__":
    # 1) Fetch history, compute indicators
    df = fetch_history()
    df = compute_indicators(df)
    # 2) Train or load model
    model = load_or_train_model(df)
    # Optional quick backtest summary
    try:
        quick_backtest(df, model)
    except Exception as e:
        print("Backtest skipped:", e)
    # 3) Enter live loop
    run_live_loop()
