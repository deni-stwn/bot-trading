import os
from typing import Dict, Any

# MT5 Account Configuration
LOGIN = 271233098
PASSWORD = "Sukabumi12@"
SERVER = "Exness-MT5Trial14"

# LOGIN = 145789224
# PASSWORD = "Sukabumi12@"
# SERVER = "Exness-MT5Real17"



# Trading Configuration
TRADING_CONFIG = {
    # "symbol": "XAUUSDm",
    "symbol": "BTCUSDm",
    "timeframe": 1,  # 5 minutes
    "max_positions": 3,
    "risk_percentage": 1.0,
    "base_lot_size": 0.01,
    "magic_number": 234000
}

# ML Configuration
ML_CONFIG = {
    "enabled": True,
    "model_path": "models/trained_models/improved_model.pkl",
    "model_retrain_interval": 168,  # hours (1 week)
    "min_training_samples": 1000,
    "prediction_confidence_threshold": 0.6,
    "ml_traditional_weight": 0.4,  # 40% ML, 60% traditional
    "feature_importance_threshold": 0.01
}

# Enhanced Strategy Configuration with ML
STRATEGY_CONFIG = {
    "min_confluences": 2,  # Reduced because of ML
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "atr_multiplier": 2.0,
    "min_time_between_signals": 180,
    "volume_threshold": 1.2,
    "ml_enabled": True,
    "ml_confidence_threshold": 0.6
}

# Strategy Configuration
# STRATEGY_CONFIG = {
#     "min_confluences": 3,
#     "rsi_oversold": 30,
#     "rsi_overbought": 70,
#     "atr_multiplier": 2.0,
#     "min_time_between_signals": 300,  # 5 minutes
#     "volume_threshold": 1.5
# }

# Strategy Configuration - KURANGI SYARAT SINYAL
# STRATEGY_CONFIG = {
#     "min_confluences": 2,        # Turunkan dari 3 ke 2
#     "rsi_oversold": 35,          # Lebih sensitif (dari 30)
#     "rsi_overbought": 65,        # Lebih sensitif (dari 70)
#     "atr_multiplier": 2.0,
#     "min_time_between_signals": 180,  # Kurangi dari 300 ke 180 detik
#     "volume_threshold": 1.2      # Turunkan dari 1.5 ke 1.2
# }

# Risk Management Configuration
RISK_CONFIG = {
    "max_daily_loss": 100.0,  # USD
    "max_drawdown": 20.0,     # Percentage
    "max_risk_per_trade": 2.0, # Percentage of balance
    "min_profit_factor": 1.2
}

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "file": "trading_bot.log",
    "max_size": 10485760,  # 10MB
    "backup_count": 5
}