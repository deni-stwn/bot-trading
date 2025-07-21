import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Optional
import logging

class ImprovedMLPredictorV2:
    def __init__(self, model_path: str = "models/trained_models/improved_model_v2.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        self.load_model()
        
    def load_model(self) -> bool:
        """Load the improved model v2"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.logger.info(f"Improved model v2 loaded from {self.model_path}")
                
                # Load feature info
                info_path = "models/trained_models/model_info_v2.pkl"
                if os.path.exists(info_path):
                    info = joblib.load(info_path)
                    self.feature_names = info.get('feature_names', [])
                    print(f"✅ Model v2 loaded with accuracy: {info.get('accuracy', 'N/A'):.4f}")
                    print(f"📊 Features: {self.feature_names}")
                    
                return True
            else:
                self.logger.warning(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def predict_signal(self, market_data: pd.DataFrame) -> Dict:
        """Predict signal with improved v2 method"""
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            if len(market_data) < 100:
                return {"error": "Insufficient data for prediction"}
            
            # Create enhanced features
            df = market_data.copy()
            
            # Basic features
            df['returns'] = df['close'].pct_change()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['volatility'] = df['returns'].rolling(20).std()
            
            # NEW FEATURES (Bollinger Bands & Price Change)
            bb_upper, bb_lower = self._bollinger_bands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['price_change'] = df['close'].pct_change(5)  # 5-period change
            
            # Get latest data
            latest = df.iloc[-1:][self.feature_names]
            
            # Handle missing values
            latest = latest.bfill().ffill().fillna(0)
            
            if latest.isnull().any().any():
                return {"error": "Unable to calculate features"}
            
            # Make prediction
            prediction = self.model.predict(latest)[0]
            probabilities = self.model.predict_proba(latest)[0]
            
            # Convert to trading signal
            if prediction == 2:  # BUY class
                signal = "BUY"
                confidence = probabilities[2]
            elif prediction == 0:  # SELL class
                signal = "SELL"
                confidence = probabilities[0]
            else:  # HOLD class
                signal = "HOLD"
                confidence = probabilities[1]
            
            return {
                "signal": signal,
                "confidence": float(confidence),
                "prediction": int(prediction),
                "probabilities": {
                    "sell": float(probabilities[0]),
                    "hold": float(probabilities[1]),
                    "buy": float(probabilities[2])
                },
                "features": latest.iloc[0].to_dict(),
                "model_version": "v2_improved"
            }
            
        except Exception as e:
            self.logger.error(f"Error in improved v2 prediction: {e}")
            return {"error": str(e)}
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower