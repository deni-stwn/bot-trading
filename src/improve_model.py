# terbaru

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

def improve_model():
    """Improve model with hyperparameter tuning"""
    print("🔧 Improving Model Performance...")
    
    # Load data
    data_path = "data/historical/BTCUSDm_5M_180days_20250528.csv"
    df = pd.read_csv(data_path)
    
    # Prepare features (same as before)
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Add more features
    df['bb_upper'], df['bb_lower'] = bollinger_bands(df['close'])
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['price_change'] = df['close'].pct_change(5)  # 5-period change
    
    # Create target
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    returns = df['future_return'].dropna()
    
    buy_threshold = returns.quantile(0.8)   # Top 20% = BUY
    sell_threshold = returns.quantile(0.2)  # Bottom 20% = SELL
    
    df['target'] = 1  # HOLD
    df.loc[df['future_return'] > buy_threshold, 'target'] = 2    # BUY
    df.loc[df['future_return'] < sell_threshold, 'target'] = 0   # SELL
    
    # Features
    feature_cols = ['returns', 'sma_20', 'sma_50', 'rsi', 'volatility', 
                   'bb_position', 'price_change']
    
    df = df.dropna()
    X = df[feature_cols]
    y = df['target']
    
    print(f"📊 Improved dataset: {len(X)} samples")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    
    # Hyperparameter tuning
    print("🔍 Hyperparameter tuning...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 10, 12, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 6]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"📊 Best CV score: {grid_search.best_score_:.4f}")
    
    # Save improved model
    joblib.dump(grid_search.best_estimator_, 'models/trained_models/improved_model_v2.pkl')
    
    feature_info = {
        'feature_names': feature_cols,
        'model_type': 'RandomForest_Tuned',
        'accuracy': grid_search.best_score_,
        'parameters': grid_search.best_params_
    }
    joblib.dump(feature_info, 'models/trained_models/model_info_v2.pkl')
    
    print("✅ Improved model saved!")

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

if __name__ == "__main__":
    improve_model()