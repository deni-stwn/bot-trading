import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Optional, Tuple
import logging
from ml.feature_engineering import FeatureEngineer

class MLPredictor:
    def __init__(self, model_path: str = "models/trained_models/best_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.logger = logging.getLogger(__name__)
        self.load_model()
        
    def load_model(self) -> bool:
        """Load trained ML model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.logger.info(f"ML model loaded from {self.model_path}")
                return True
            else:
                self.logger.warning(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def predict_signal(self, market_data: pd.DataFrame) -> Dict:
        """Predict trading signal using ML model"""
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            if len(market_data) < 100:
                return {"error": "Insufficient data for prediction"}
            
            # Engineer features
            features_df = self.feature_engineer.create_features(market_data)
            
            # Get the latest data point
            latest_features = features_df.iloc[-1:].copy()
            
            # Select feature columns (same as training)
            feature_cols = [col for col in features_df.columns 
                           if col not in ['time', 'open', 'high', 'low', 'close', 
                                         'tick_volume', 'volume']]
            
            X = latest_features[feature_cols]
            
            # Handle missing values
            X = X.fillna(method='bfill').fillna(method='ffill')
            
            if X.isna().any().any():
                return {"error": "Features contain NaN values"}
            
            # Scale features
            X_scaled = self.feature_engineer.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            # Convert prediction to signal
            if prediction == 1:
                signal = "BUY"
                confidence = prediction_proba[np.where(self.model.classes_ == 1)[0][0]]
            elif prediction == -1:
                signal = "SELL"
                confidence = prediction_proba[np.where(self.model.classes_ == -1)[0][0]]
            else:
                signal = "HOLD"
                confidence = prediction_proba[np.where(self.model.classes_ == 0)[0][0]]
            
            return {
                "signal": signal,
                "confidence": float(confidence),
                "prediction": int(prediction),
                "probabilities": {
                    "buy": float(prediction_proba[np.where(self.model.classes_ == 1)[0][0]] if 1 in self.model.classes_ else 0),
                    "sell": float(prediction_proba[np.where(self.model.classes_ == -1)[0][0]] if -1 in self.model.classes_ else 0),
                    "hold": float(prediction_proba[np.where(self.model.classes_ == 0)[0][0]] if 0 in self.model.classes_ else 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the loaded model"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_engineer.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        return None