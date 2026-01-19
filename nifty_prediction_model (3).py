"""
INSTITUTIONAL-GRADE NIFTY TRADING SYSTEM
========================================
Advanced AI Trading Model with Institutional-Level Features
- Ensemble of multiple deep learning architectures
- Alternative data integration (VIX, sectoral indices, global markets)
- Advanced risk management and position sizing
- Real-time market microstructure analysis
- Regime detection and adaptive modeling
- Monte Carlo simulations for risk assessment

Requirements:
pip install tensorflow yfinance pandas numpy matplotlib seaborn scikit-learn ta requests beautifulsoup4 plotly dash xgboost lightgbm catboost scipy statsmodels arch

Author: AI Trading Systems
Version: Professional 2.0
"""

import os
import sys
import warnings
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from pathlib import Path
import joblib

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union

# ML and DL libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
from arch import arch_model

# Advanced ML
try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    HAS_GRADIENT_BOOSTING = True
except ImportError:
    HAS_GRADIENT_BOOSTING = False
    print("⚠️ Advanced gradient boosting libraries not found. Installing...")
    os.system("pip install xgboost lightgbm catboost")

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
                                   Input, concatenate, Attention, LayerNormalization,
                                   MultiHeadAttention, GlobalAveragePooling1D, Add,
                                   BatchNormalization, GRU, Bidirectional)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

# Data sources
import yfinance as yf
import ta

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class AdvancedMarketDataFetcher:
    """
    Professional-grade market data fetcher with multiple sources
    """
    
    def __init__(self):
        self.data_sources = {
            'nifty': ['^NSEI', 'NIFTY_50.NS', '^NSEBANK'],
            'vix': ['^NSEVIXI', 'INDIAVIX.NS'],
            'sectoral': [
                '^NSEBANK', '^NSEIT', '^NSEFMCG', '^NSEPHARMA',
                '^NSEMETAL', '^NSEAUTO', '^NSEREAL', '^NSEENERGY'
            ],
            'global': ['^GSPC', '^IXIC', '^DJI', '^FTSE', '^N225', '^HSI'],
            'commodities': ['GC=F', 'CL=F', '^TNX'],  # Gold, Oil, 10Y Treasury
            'currency': ['USDINR=X', 'DX-Y.NYB']  # USD/INR, Dollar Index
        }
    
    def fetch_comprehensive_data(self, period="3y", interval="1d"):
        """
        Fetch comprehensive market data from multiple sources
        """
        print("🌐 Fetching comprehensive market data...")
        all_data = {}
        
        def fetch_symbol_data(symbol, category):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                if len(data) > 50:
                    return symbol, category, data
            except Exception as e:
                print(f"⚠️ Failed to fetch {symbol}: {e}")
            return None, None, None
        
        # Parallel data fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for category, symbols in self.data_sources.items():
                for symbol in symbols:
                    future = executor.submit(fetch_symbol_data, symbol, category)
                    futures.append(future)
            
            for future in as_completed(futures):
                symbol, category, data = future.result()
                if data is not None:
                    if category not in all_data:
                        all_data[category] = {}
                    all_data[category][symbol] = data
                    print(f"✅ Fetched {symbol} ({category}): {len(data)} points")
        
        return all_data
    
    def create_market_regime_features(self, data_dict):
        """
        Create advanced market regime and cross-asset features
        """
        print("🔬 Creating market regime features...")
        
        # Get primary Nifty data
        nifty_data = None
        for symbol, data in data_dict.get('nifty', {}).items():
            if len(data) > 0:
                nifty_data = data.copy()
                break
        
        if nifty_data is None:
            print("❌ No Nifty data found")
            return None
        
        # Add VIX data
        vix_data = None
        for symbol, data in data_dict.get('vix', {}).items():
            if len(data) > 0:
                vix_data = data['Close'].reindex(nifty_data.index, method='ffill')
                nifty_data['VIX'] = vix_data
                break
        
        # Add sectoral strength indicators
        sectoral_data = data_dict.get('sectoral', {})
        for i, (symbol, data) in enumerate(sectoral_data.items()):
            if len(data) > 0:
                sector_returns = data['Close'].pct_change().reindex(nifty_data.index, method='ffill')
                nifty_data[f'Sector_{i}_Returns'] = sector_returns
                
                # Relative strength
                nifty_returns = nifty_data['Close'].pct_change()
                relative_strength = sector_returns - nifty_returns
                nifty_data[f'Sector_{i}_RS'] = relative_strength.rolling(20).mean()
        
        # Add global market indicators
        global_data = data_dict.get('global', {})
        for i, (symbol, data) in enumerate(global_data.items()):
            if len(data) > 0:
                # Time-shifted for overnight effects
                global_close = data['Close'].reindex(nifty_data.index, method='ffill')
                global_returns = global_close.pct_change()
                nifty_data[f'Global_{i}_Returns'] = global_returns
                
                # Correlation regime
                correlation = nifty_data['Close'].pct_change().rolling(60).corr(global_returns)
                nifty_data[f'Global_{i}_Corr'] = correlation
        
        # Add commodity and currency effects
        commodities_data = data_dict.get('commodities', {})
        for i, (symbol, data) in enumerate(commodities_data.items()):
            if len(data) > 0:
                commodity_returns = data['Close'].pct_change().reindex(nifty_data.index, method='ffill')
                nifty_data[f'Commodity_{i}_Returns'] = commodity_returns
        
        currency_data = data_dict.get('currency', {})
        for i, (symbol, data) in enumerate(currency_data.items()):
            if len(data) > 0:
                currency_returns = data['Close'].pct_change().reindex(nifty_data.index, method='ffill')
                nifty_data[f'Currency_{i}_Returns'] = currency_returns
        
        return nifty_data

class AdvancedFeatureEngine:
    """
    Institutional-grade feature engineering
    """
    
    @staticmethod
    def calculate_microstructure_features(data):
        """
        Calculate market microstructure features
        """
        print("🔬 Calculating microstructure features...")
        
        df = data.copy()
        
        # Price efficiency measures
        df['Amihud_Illiquidity'] = np.abs(df['Close'].pct_change()) / (df['Volume'] + 1e-10)
        df['Price_Impact'] = df['Amihud_Illiquidity'].rolling(20).mean()
        
        # Intraday patterns
        df['OHLC_Average'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Weighted_Price'] = (df['High'] + df['Low'] + 2*df['Close']) / 4
        
        # Gap analysis
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Fill'] = np.where(
            df['Gap'] > 0,
            np.minimum(df['Low'], df['Close'].shift(1)) / df['Open'],
            np.maximum(df['High'], df['Close'].shift(1)) / df['Open']
        )
        
        # Volume-price relationship
        df['Volume_Price_Trend'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        df['Ease_of_Movement'] = ta.volume.ease_of_movement(df['High'], df['Low'], df['Volume'])
        df['Money_Flow_Index'] = ta.volume.money_flow_index(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        return df
    
    @staticmethod
    def calculate_regime_features(data):
        """
        Calculate market regime detection features
        """
        print("📊 Calculating regime features...")
        
        df = data.copy()
        returns = df['Close'].pct_change()
        
        # Volatility regimes
        df['Realized_Vol'] = returns.rolling(20).std() * np.sqrt(252)
        df['GARCH_Vol'] = AdvancedFeatureEngine._calculate_garch_volatility(returns)
        df['Vol_Regime'] = pd.cut(df['Realized_Vol'], bins=3, labels=[0, 1, 2])
        
        # Trend regimes using regime-switching models
        df['Trend_Strength'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        df['Trend_Direction'] = np.where(
            df['Close'] > df['Close'].rolling(50).mean(), 1, -1
        )
        
        # Market stress indicators
        df['Stress_Indicator'] = (
            df['Realized_Vol'].rolling(20).rank(pct=True) * 0.4 +
            df['VIX'].rolling(20).rank(pct=True) * 0.6
        ) if 'VIX' in df.columns else df['Realized_Vol'].rolling(20).rank(pct=True)
        
        # Momentum regimes
        momentum_periods = [5, 10, 20, 50]
        momentum_scores = []
        for period in momentum_periods:
            momentum = (df['Close'] / df['Close'].shift(period) - 1) * 100
            momentum_scores.append(momentum)
        
        df['Momentum_Composite'] = pd.concat(momentum_scores, axis=1).mean(axis=1)
        df['Momentum_Regime'] = pd.cut(df['Momentum_Composite'], bins=3, labels=[-1, 0, 1])
        
        return df
    
    @staticmethod
    def _calculate_garch_volatility(returns, window=252):
        """
        Calculate GARCH volatility
        """
        garch_vol = []
        returns_clean = returns.dropna()
        
        for i in range(window, len(returns_clean)):
            try:
                data_window = returns_clean.iloc[i-window:i] * 100  # Convert to percentage
                model = arch_model(data_window, vol='Garch', p=1, q=1)
                result = model.fit(disp='off')
                forecast = result.forecast(horizon=1)
                vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100  # Convert back
                garch_vol.append(vol)
            except:
                garch_vol.append(np.nan)
        
        # Pad with NaNs for alignment
        full_garch_vol = [np.nan] * window + garch_vol
        return pd.Series(full_garch_vol, index=returns.index)
    
    @staticmethod
    def calculate_alternative_features(data):
        """
        Calculate alternative and proprietary features
        """
        print("🧠 Calculating alternative features...")
        
        df = data.copy()
        
        # Fractal dimension
        df['Fractal_Dimension'] = AdvancedFeatureEngine._calculate_fractal_dimension(df['Close'])
        
        # Hurst exponent for mean reversion
        df['Hurst_Exponent'] = AdvancedFeatureEngine._calculate_hurst_exponent(df['Close'])
        
        # Entropy measures
        df['Sample_Entropy'] = AdvancedFeatureEngine._calculate_sample_entropy(df['Close'])
        
        # Lyapunov exponent for chaos
        df['Lyapunov_Exponent'] = AdvancedFeatureEngine._calculate_lyapunov_exponent(df['Close'])
        
        # Advanced statistical features
        returns = df['Close'].pct_change()
        
        # Higher moments
        for window in [20, 60]:
            df[f'Skewness_{window}'] = returns.rolling(window).skew()
            df[f'Kurtosis_{window}'] = returns.rolling(window).kurt()
            df[f'Jarque_Bera_{window}'] = returns.rolling(window).apply(
                lambda x: stats.jarque_bera(x.dropna())[0] if len(x.dropna()) > 10 else np.nan
            )
        
        # Regime-dependent correlations with multiple timeframes
        for lag in [1, 5, 20]:
            df[f'Autocorr_{lag}'] = returns.rolling(60).apply(
                lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag + 10 else np.nan
            )
        
        return df
    
    @staticmethod
    def _calculate_fractal_dimension(prices, window=20):
        """Calculate Higuchi fractal dimension"""
        def higuchi_fd(X, k_max=10):
            N = len(X)
            L = np.zeros(k_max)
            
            for k in range(1, k_max + 1):
                Lk = []
                for m in range(k):
                    Lm = 0
                    for i in range(1, int((N - m) / k)):
                        Lm += abs(X[m + i * k] - X[m + (i - 1) * k])
                    Lm = Lm * (N - 1) / (k * int((N - m) / k) * k)
                    Lk.append(Lm)
                L[k - 1] = np.mean(Lk)
            
            # Linear regression to find slope
            k = np.arange(1, k_max + 1)
            coeffs = np.polyfit(np.log(k), np.log(L), 1)
            return -coeffs[0]
        
        fractal_dims = []
        for i in range(window, len(prices)):
            try:
                fd = higuchi_fd(prices.iloc[i-window:i].values)
                fractal_dims.append(fd)
            except:
                fractal_dims.append(np.nan)
        
        return pd.Series([np.nan] * window + fractal_dims, index=prices.index)
    
    @staticmethod
    def _calculate_hurst_exponent(prices, window=100):
        """Calculate Hurst exponent using R/S analysis"""
        def hurst_rs(X):
            N = len(X)
            if N < 20:
                return 0.5
            
            # Calculate log returns
            Y = np.diff(np.log(X))
            
            # Calculate R/S for different lag values
            lags = np.unique(np.logspace(0.7, np.log10(N/2), 15).astype(int))
            RS = []
            
            for lag in lags:
                # Split the series
                n_splits = N // lag
                rs_values = []
                
                for i in range(n_splits):
                    start_idx = i * lag
                    end_idx = start_idx + lag
                    series = Y[start_idx:end_idx]
                    
                    if len(series) == 0:
                        continue
                    
                    # Calculate mean
                    mean_series = np.mean(series)
                    
                    # Calculate cumulative deviations
                    cumulative_deviations = np.cumsum(series - mean_series)
                    
                    # Calculate range
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    
                    # Calculate standard deviation
                    S = np.std(series)
                    
                    # Calculate R/S
                    if S != 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    RS.append(np.mean(rs_values))
            
            if len(RS) > 1:
                # Linear regression
                coeffs = np.polyfit(np.log(lags[:len(RS)]), np.log(RS), 1)
                return coeffs[0]
            return 0.5
        
        hurst_values = []
        for i in range(window, len(prices)):
            try:
                hurst = hurst_rs(prices.iloc[i-window:i].values)
                hurst_values.append(hurst)
            except:
                hurst_values.append(0.5)
        
        return pd.Series([0.5] * window + hurst_values, index=prices.index)
    
    @staticmethod
    def _calculate_sample_entropy(prices, window=50, m=2, r_factor=0.2):
        """Calculate Sample Entropy"""
        def sample_entropy_calc(U, m, r):
            N = len(U)
            
            def _maxdist(xi, xj):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                X = np.array([U[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template_i = X[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, X[j]) <= r:
                            C[i] += 1.0
                
                return (C.sum() - N + m - 1.0) / ((N - m + 1.0) * (N - m))
            
            return -np.log(_phi(m + 1) / _phi(m))
        
        entropy_values = []
        for i in range(window, len(prices)):
            try:
                data = prices.iloc[i-window:i].values
                r = r_factor * np.std(data)
                entropy = sample_entropy_calc(data, m, r)
                if np.isfinite(entropy):
                    entropy_values.append(entropy)
                else:
                    entropy_values.append(np.nan)
            except:
                entropy_values.append(np.nan)
        
        return pd.Series([np.nan] * window + entropy_values, index=prices.index)
    
    @staticmethod
    def _calculate_lyapunov_exponent(prices, window=100):
        """Calculate largest Lyapunov exponent"""
        def lyapunov_exp(data, tau=1, n=10):
            N = len(data)
            if N < n + tau:
                return np.nan
            
            # Embed the data
            Y = np.zeros((N - (n - 1) * tau, n))
            for i in range(n):
                Y[:, i] = data[(n - 1 - i) * tau:N - i * tau]
            
            # Find nearest neighbors
            lyap_sum = 0
            count = 0
            
            for i in range(len(Y) - 1):
                distances = np.sqrt(np.sum((Y - Y[i]) ** 2, axis=1))
                distances[i] = np.inf  # Exclude self
                
                if np.min(distances) > 0:
                    nearest_idx = np.argmin(distances)
                    
                    # Calculate divergence
                    if i + 1 < len(Y) and nearest_idx + 1 < len(Y):
                        d0 = distances[nearest_idx]
                        d1 = np.sqrt(np.sum((Y[i + 1] - Y[nearest_idx + 1]) ** 2))
                        
                        if d0 > 0 and d1 > 0:
                            lyap_sum += np.log(d1 / d0)
                            count += 1
            
            return lyap_sum / count if count > 0 else np.nan
        
        lyap_values = []
        for i in range(window, len(prices)):
            try:
                lyap = lyapunov_exp(prices.iloc[i-window:i].values)
                lyap_values.append(lyap)
            except:
                lyap_values.append(np.nan)
        
        return pd.Series([np.nan] * window + lyap_values, index=prices.index)

class InstitutionalEnsembleModel:
    """
    Institutional-grade ensemble model combining multiple architectures
    """
    
    def __init__(self, lookback_period=120, prediction_horizons=[1, 3, 5], confidence_level=0.95):
        self.lookback_period = lookback_period
        self.prediction_horizons = prediction_horizons
        self.confidence_level = confidence_level
        
        # Scalers
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler()
        
        # Model components
        self.models = {}
        self.model_weights = {}
        self.feature_importance = {}
        
        # Risk management
        self.risk_manager = None
        self.position_sizer = None
        
        print("🏛️ Institutional Ensemble Model initialized")
    
    def build_transformer_model(self, input_shape, output_shape):
        """
        Build Transformer-based model for time series
        """
        inputs = Input(shape=input_shape)
        
        # Multi-head attention layers
        attention1 = MultiHeadAttention(
            num_heads=8, 
            key_dim=64,
            dropout=0.1
        )(inputs, inputs)
        attention1 = LayerNormalization()(inputs + attention1)
        
        attention2 = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1
        )(attention1, attention1)
        attention2 = LayerNormalization()(attention1 + attention2)
        
        # Global pooling and dense layers
        pooled = GlobalAveragePooling1D()(attention2)
        
        dense1 = Dense(256, activation='relu')(pooled)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.2)(dense2)
        
        outputs = Dense(output_shape, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss='huber',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape, output_shape):
        """
        Build advanced CNN-LSTM hybrid model
        """
        inputs = Input(shape=input_shape)
        
        # Multi-scale CNN features
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = Conv1D(128, 5, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.2)(conv2)
        pool1 = MaxPooling1D(2)(conv2)
        
        conv3 = Conv1D(256, 3, activation='relu', padding='same')(pool1)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(0.2)(conv3)
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(conv3)
        lstm2 = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(lstm1)
        
        # Dense layers with regularization
        dense1 = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(lstm2)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(dense1)
        dense2 = Dropout(0.3)(dense2)
        
        outputs = Dense(output_shape, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_wavenet_model(self, input_shape, output_shape):
        """
        Build WaveNet-inspired model for time series
        """
        inputs = Input(shape=input_shape)
        
        # Initial convolution
        x = Conv1D(32, 2, padding='causal', activation='relu')(inputs)
        
        # Dilated convolutions
        skip_connections = []
        for dilation_rate in [1, 2, 4, 8, 16, 32]:
            # Dilated convolution
            conv = Conv1D(
                64, 2, 
                padding='causal',
                dilation_rate=dilation_rate,
                activation='tanh'
            )(x)
            
            # Gated activation
            gate = Conv1D(
                64, 2,
                padding='causal', 
                dilation_rate=dilation_rate,
                activation='sigmoid'
            )(x)
            
            z = tf.multiply(conv, gate)
            
            # Skip connection
            skip = Conv1D(32, 1)(z)
            skip_connections.append(skip)
            
            # Residual connection
            residual = Conv1D(32, 1)(z)
            x = Add()([x, residual])
        
        # Combine skip connections
        skip_sum = Add()(skip_connections)
        
        # Final layers
        x = Conv1D(128, 1, activation='relu')(skip_sum)
        x = Conv1D(256, 1, activation='relu')(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(output_shape, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val, epochs=100):
        """
        Train ensemble of different model architectures
        """
        print("🎯 Training ensemble models...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Model architectures to train
        model_configs = {
            'transformer': self.build_transformer_model,
            'cnn_lstm': self.build_cnn_lstm_model,
            'wavenet': self.build_wavenet_model
        }
        
        trained_models = {}
        model_scores = {}
        
        for name, build_func in model_configs.items():
            print(f"🔧 Training {name} model...")
            
            try:
                # Build model
                model = build_func(input_shape, output_shape)
                
                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=25,
                        restore_best_weights=True,
                        verbose=0
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=15,
                        min_lr=0.0001,
                        verbose=0
                    )
                ]
                
                # Train model
                history = model.fit(
                   X_train, y_train,
                  validation_data=(X_val, y_val),
                 epochs=epochs,
                  batch_size=64,
                  callbacks=callbacks,
                 verbose=2
                 )
            except Exception as e:
                print(f"❌ Error training {name} model: {e}")