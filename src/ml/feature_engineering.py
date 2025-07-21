import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for ML model using custom implementations"""
        features_df = df.copy()
        
        # 1. TECHNICAL INDICATORS (Custom implementations)
        features_df = self._add_technical_indicators(features_df)
        
        # 2. PRICE PATTERNS
        features_df = self._add_price_patterns(features_df)
        
        # 3. STATISTICAL FEATURES
        features_df = self._add_statistical_features(features_df)
        
        # 4. TIME-BASED FEATURES
        features_df = self._add_time_features(features_df)
        
        # 5. MARKET MICROSTRUCTURE
        features_df = self._add_microstructure_features(features_df)
        
        # 6. VOLATILITY FEATURES
        features_df = self._add_volatility_features(features_df)
        
        return features_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using custom implementations (no TA-Lib)"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df.get('tick_volume', df.get('volume', np.ones(len(df)))).values
        
        # Simple Moving Averages
        df['sma_5'] = self._sma(close, 5)
        df['sma_10'] = self._sma(close, 10)
        df['sma_20'] = self._sma(close, 20)
        df['sma_50'] = self._sma(close, 50)
        
        # Exponential Moving Averages
        df['ema_9'] = self._ema(close, 9)
        df['ema_21'] = self._ema(close, 21)
        
        # RSI
        df['rsi_14'] = self._rsi(close, 14)
        df['rsi_7'] = self._rsi(close, 7)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._stochastic(high, low, close, 14, 3)
        
        # Williams %R
        df['williams_r'] = self._williams_r(high, low, close, 14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._macd(close, 12, 26, 9)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._bollinger_bands(close, 20, 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = self._atr(high, low, close, 14)
        
        # Volume indicators
        df['volume_sma'] = self._sma(volume, 20)
        df['volume_ratio'] = volume / df['volume_sma']
        
        return df
    
    def _sma(self, data: np.ndarray, period: int) -> pd.Series:
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period, min_periods=1).mean()
    
    def _ema(self, data: np.ndarray, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period, min_periods=1).mean()
    
    def _rsi(self, data: np.ndarray, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        close = pd.Series(data)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def _stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        
        lowest_low = low_s.rolling(window=k_period, min_periods=1).min()
        highest_high = high_s.rolling(window=k_period, min_periods=1).max()
        
        k_percent = 100 * ((close_s - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
        
        return k_percent.fillna(50), d_percent.fillna(50)
    
    def _williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                    period: int = 14) -> pd.Series:
        """Williams %R"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        
        highest_high = high_s.rolling(window=period, min_periods=1).max()
        lowest_low = low_s.rolling(window=period, min_periods=1).min()
        
        wr = -100 * ((highest_high - close_s) / (highest_high - lowest_low))
        return wr.fillna(-50)
    
    def _macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, 
              signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        close = pd.Series(data)
        ema_fast = close.ewm(span=fast, min_periods=1).mean()
        ema_slow = close.ewm(span=slow, min_periods=1).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, min_periods=1).mean()
        macd_histogram = macd_line - macd_signal
        
        return macd_line.fillna(0), macd_signal.fillna(0), macd_histogram.fillna(0)
    
    def _bollinger_bands(self, data: np.ndarray, period: int = 20, 
                        std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        close = pd.Series(data)
        sma = close.rolling(window=period, min_periods=1).mean()
        std = close.rolling(window=period, min_periods=1).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band.fillna(close), sma.fillna(close), lower_band.fillna(close)
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
             period: int = 14) -> pd.Series:
        """Average True Range"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        
        tr1 = high_s - low_s
        tr2 = abs(high_s - close_s.shift())
        tr3 = abs(low_s - close_s.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr.fillna(tr.mean())
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        # Price relative to moving averages
        df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        df['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        
        # MA crossovers
        df['sma_cross_5_20'] = ((df['sma_5'] > df['sma_20']) & 
                               (df['sma_5'].shift(1) <= df['sma_20'].shift(1))).astype(int)
        
        # Price action patterns
        df['higher_high'] = ((df['high'] > df['high'].shift(1)) & 
                            (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) & 
                          (df['low'].shift(1) < df['low'].shift(2))).astype(int)
        
        # Candlestick patterns
        df['doji'] = (abs(df['close'] - df['open']) <= 
                     (df['high'] - df['low']) * 0.1).astype(int)
        
        body_size = abs(df['close'] - df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        
        df['hammer'] = ((df['close'] > df['open']) & 
                       (lower_shadow > 2 * body_size) &
                       (upper_shadow < body_size)).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window, min_periods=1).std()
            df[f'rolling_skew_{window}'] = df['close'].rolling(window, min_periods=1).skew()
            df[f'rolling_kurt_{window}'] = df['close'].rolling(window, min_periods=1).kurt()
            
        # Price percentiles
        df['price_percentile_20'] = df['close'].rolling(20, min_periods=1).rank(pct=True)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Spread
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['close']
        
        # Body vs shadow
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
        
        # Normalized features
        total_range = df['spread']
        df['body_pct'] = df['body_size'] / total_range
        df['upper_shadow_pct'] = df['upper_shadow'] / total_range
        df['lower_shadow_pct'] = df['lower_shadow'] / total_range
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Realized volatility
        df['volatility_5'] = df['returns'].rolling(5, min_periods=1).std() * np.sqrt(5)
        df['volatility_20'] = df['returns'].rolling(20, min_periods=1).std() * np.sqrt(20)
        
        # Volatility ratio
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        return df
    
    def prepare_ml_dataset(self, df: pd.DataFrame, 
                          target_periods: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare dataset for ML training"""
        try:
            print(f"🔧 Preparing ML dataset from {len(df)} data points...")
            
            # Create features
            features_df = self.create_features(df)
            print(f"✅ Features created: {features_df.shape[1]} columns")
            
            # Create target variable (future returns)
            features_df['future_return'] = features_df['close'].shift(-target_periods) / features_df['close'] - 1
            
            # Create classification target
            features_df['target'] = 0  # Hold
            features_df.loc[features_df['future_return'] > 0.01, 'target'] = 1    # Buy (>1% return)
            features_df.loc[features_df['future_return'] < -0.01, 'target'] = -1  # Sell (<-1% return)
            
            # Select feature columns
            exclude_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'volume', 
                           'real_volume', 'future_return', 'target']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            print(f"📊 Selected {len(feature_cols)} features for training")
            
            X = features_df[feature_cols].copy()
            y = features_df['target'].copy()
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Remove rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            print(f"📊 Clean dataset: {len(X)} samples after removing NaN")
            
            if len(X) == 0:
                print("❌ No valid samples after cleaning")
                return None, None
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            self.feature_names = list(X.columns)
            
            print(f"✅ Dataset prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
            print(f"📊 Target distribution: {y.value_counts().to_dict()}")
            
            return X_scaled, y
            
        except Exception as e:
            print(f"❌ Error preparing ML dataset: {e}")
            return None, None