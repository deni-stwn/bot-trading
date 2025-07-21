from ml.data_collector import DataCollector
from ml.feature_engineering import FeatureEngineer
from ml.model_trainer import MLModelTrainer
from config.settings import LOGIN, PASSWORD, SERVER, TRADING_CONFIG

# Initialize collector
collector = DataCollector(LOGIN, PASSWORD, SERVER)

# Collect 1 year of BTCUSD data
training_data = collector.collect_training_data(
    symbol="BTCUSDm",
    days_back=365,
    timeframe=5
)

# Update existing data with new bars
collector.update_historical_data("BTCUSDm", timeframe=5)

# Collect data for multiple trading pairs
symbols = ["BTCUSDm", "XAUUSDm", "EURUSDm"]
multi_data = collector.collect_multiple_symbols(symbols, days_back=180)

def train_model():
    print("🤖 Training ML Model for Trading Bot")
    print("=" * 40)
    
    # Collect data
    collector = DataCollector(LOGIN, PASSWORD, SERVER)
    historical_data = collector.collect_training_data(
        symbol=TRADING_CONFIG["symbol"],
        days_back=365  # 1 year of data
    )
    
    if historical_data is None:
        print("❌ Failed to collect training data")
        return
    
    print(f"✅ Collected {len(historical_data)} data points")
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.prepare_ml_dataset(historical_data)
    
    print(f"✅ Features engineered: {X.shape[1]} features")
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    
    # Train models
    trainer = MLModelTrainer()
    results = trainer.train_models(X, y)
    
    print("\n📊 Training Results:")
    for name, result in results.items():
        print(f"{name}: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
    
    # Hyperparameter tuning for best model
    best_model = trainer.hyperparameter_tuning(X, y, 'xgboost')
    
    print("✅ ML Model training completed!")
    print("You can now run the bot with ML predictions enabled.")

if __name__ == "__main__":
    train_model()